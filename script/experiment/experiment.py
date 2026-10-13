
"""
Run experiments and save results in resumable chunks.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor

from config import PROJECT_ROOT, SLIP_NORM_MEAN, SLIP_NORM_STD, save_config_snapshot
from script.experiment.design_matrix import ensure_design_matrix
from script.experiment.constituent_embedding import (
    ensure_constituent_embeddings,
    make_composite,
)

sys.path.insert(0, str(PROJECT_ROOT / "external" / "SLIP"))
import models


# %% Runtime settings
BATCH_SIZES = {"small": 32, "base": 16, "large": 4}
CHUNK_SIZE = 5000


# %% Model loading
def load_model(model_id: str, device: torch.device):
    """Load a pretrained SLIP-family model."""
    checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "fb_slip" / f"{model_id}.pt",
        map_location="cpu",
    )
    args = checkpoint["args"]

    model = getattr(models, args.model)(
        rand_embed=False,
        ssl_mlp_dim=args.ssl_mlp_dim,
        ssl_emb_dim=args.ssl_emb_dim,
    )

    state_dict = {
        k.replace("module.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


# %% Attention hooks
def register_attention_hooks(model):
    """Capture attention matrices from ViT layers."""
    cache = []

    def hook(module, inputs, output):
        cache.append(output.detach())

    handles = [
        block.attn.attn_drop.register_forward_hook(hook)
        for block in model.visual.blocks
    ]
    return cache, handles


# %% Attention computation
def _aggregate_heads(attention: torch.Tensor, method: str):
    """Aggregate attention across heads."""
    if method == "mean":
        return attention.mean(dim=-3)
    if method == "max":
        return attention.max(dim=-3).values
    if method == "min":
        return attention.min(dim=-3).values


def _discard_attention(attention: torch.Tensor, ratio: float):
    """Discard the lowest attention values."""
    flat = attention.flatten(start_dim=2)
    n_discard = int(flat.shape[-1] * ratio)

    if n_discard:
        indices = flat.argsort(dim=-1)[..., :n_discard]
        flat.scatter_(-1, indices, 0)

    return flat.reshape_as(attention)


def _add_residual_and_normalize(attention: torch.Tensor):
    """Add residual connections and normalize attention rows."""
    n_tokens = attention.shape[-1]
    identity = torch.eye(
        n_tokens, device=attention.device, dtype=attention.dtype
    ).view(1, 1, n_tokens, n_tokens)

    attention = attention + identity
    return attention / attention.sum(dim=-1, keepdim=True)


def _get_patch_indices(layout: dict, patch_size: int):
    """Convert constituent regions to patch indices."""
    grid_width = layout["canvas_size"][0] // patch_size
    item_w, item_h = layout["item_size"]

    return [
        [
            row * grid_width + col
            for row in range(y // patch_size, (y + item_h) // patch_size)
            for col in range(x // patch_size, (x + item_w) // patch_size)
        ]
        for x, y in layout["positions"]
    ]
    

def attention_rollout(attentions: torch.Tensor, config: dict):
    """Compute attention rollout across transformer layers."""
    attention = _aggregate_heads(attentions, config["head_aggregation"])
    attention = _discard_attention(attention, config["discard_ratio"])
    attention = _add_residual_and_normalize(attention)

    rollout = attention[:, 0]
    for layer in range(1, attention.shape[1]):
        rollout = attention[:, layer] @ rollout

    return rollout


def attention_layer(attentions: torch.Tensor, config: dict):
    """Extract attention from a single transformer layer."""
    return _aggregate_heads(
        attentions[:, config["layer"]],
        config["head_aggregation"],
    )


def _summarize_attention(
    attention: torch.Tensor,
    patch_indices: list,
    method: str,
):
    """Summarize CLS attention within constituent regions."""
    patch_attention = attention[:, 0, 1:]
    values = []

    for indices in patch_indices:
        region = patch_attention[:, indices]

        if method == "mean":
            value = region.mean(dim=1)
        elif method == "max":
            value = region.max(dim=1).values
        elif method == "min":
            value = region.min(dim=1).values

        values.append(value)

    return torch.stack(values, dim=1)


# %% Embedding computation
def embedding_similarity(
    composite_embeddings: torch.Tensor,
    constituent_embeddings: list[torch.Tensor],
):
    """Compute composite-constituent cosine similarities."""
    return torch.stack([
        F.cosine_similarity(composite_embeddings, item, dim=1)
        for item in constituent_embeddings
    ], dim=1)


# %% Metric extraction
def _build_metric_df(trial_ids, values, metric, config_key):
    """Build long-form trial-level results."""
    n_items = values.shape[1]

    return pd.DataFrame({
        "Trial_id": trial_ids.repeat(n_items).to_numpy(),
        "Item": list(range(n_items)) * len(trial_ids),
        "Metric": metric,
        "Config_key": config_key,
        "Value": values.detach().cpu().numpy().reshape(-1),
    })


def compute_metrics(
    batch_df: pd.DataFrame,
    composite_embeddings: torch.Tensor,
    attentions: torch.Tensor,
    metrics: dict,
    constituent_embeddings: dict,
    patch_indices: list,
    constituent_cols: list,
) -> list[pd.DataFrame]:
    """Compute all configured metrics for one batch."""
    results = []
    trial_ids = batch_df["Trial_id"]

    if "embedding_similarity" in metrics:
        config = metrics["embedding_similarity"]
        items = [
            torch.stack([
                constituent_embeddings[image_id][position]
                for image_id in batch_df[col]
            ])
            for position, col in enumerate(constituent_cols)
        ]
        values = embedding_similarity(composite_embeddings, items)
        results.append(_build_metric_df(
            trial_ids, values, "embedding_similarity", config["present"]
        ))

    if "attention_rollout" in metrics:
        config = metrics["attention_rollout"]
        attention = attention_rollout(attentions, config)
        values = _summarize_attention(
            attention, patch_indices, config["region_summary"]
        )
        key = (
            f"{config['head_aggregation']}_"
            f"{config['discard_ratio']}_"
            f"{config['region_summary']}"
        )
        results.append(_build_metric_df(
            trial_ids, values, "attention_rollout", key
        ))

    if "attention_layer" in metrics:
        config = metrics["attention_layer"]
        attention = attention_layer(attentions, config)
        values = _summarize_attention(
            attention, patch_indices, config["region_summary"]
        )
        layer = "last" if config["layer"] == -1 else str(config["layer"])
        key = (
            f"{layer}_{config['head_aggregation']}_"
            f"{config['region_summary']}"
        )
        results.append(_build_metric_df(
            trial_ids, values, "attention_layer", key
        ))

    return results


# %% Experiment execution
def run_experiment(experiment: dict) -> Path:
    """Run an experiment with chunk-based saving and resumption."""
    model_id = experiment["model"]
    imageset = experiment["imageset"]
    layout = experiment["layout"]
    metrics = experiment["metrics"]

    result_dir = (
        PROJECT_ROOT / "results"
        / experiment["protocol_id"] / experiment["experiment_id"]
    )
    chunk_dir = result_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Configuration snapshot
    config_path = result_dir / "config.yml"
    if config_path.exists():
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != experiment:
            raise ValueError(f"Experiment configuration changed: {config_path}")
    else:
        save_config_snapshot(experiment, config_path)

    result_path = result_dir / "results.csv"
    if result_path.exists():
        print(f"Already completed: {experiment['experiment_id']}")
        return result_dir

    # Design matrix and model
    design_matrix, _ = ensure_design_matrix(experiment)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    batch_size = BATCH_SIZES[model_id.split("_")[1]]
    model = load_model(model_id, device)

    print(f"Experiment: {experiment['experiment_id']}")
    print(f"Device: {device} | Batch size: {batch_size}")

    # Constituent embeddings
    constituent_embeddings = {}
    if "embedding_similarity" in metrics:
        constituent_embeddings = {
            image_id: embedding.to(device)
            for image_id, embedding in ensure_constituent_embeddings(
                experiment, model, device, batch_size=batch_size
            ).items()
        }

    # Image cache
    constituent_cols = [
        f"Constituent_{i}"
        for i in range(experiment["trial_design"]["n_items_per_trial"])
    ]
    image_ids = pd.unique(design_matrix[constituent_cols].values.ravel())
    image_dir = PROJECT_ROOT / "stimuli" / imageset / "images"

    image_cache = {}
    for image_id in image_ids:
        with Image.open(image_dir / f"{image_id}.jpg") as image:
            image_cache[image_id] = pil_to_tensor(image.convert("RGB"))

    mean = torch.tensor(SLIP_NORM_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(SLIP_NORM_STD, device=device).view(1, 3, 1, 1)

    # Attention setup
    use_attention = any(
        m in metrics for m in ("attention_rollout", "attention_layer")
    )
    attention_cache, handles = (
        register_attention_hooks(model) if use_attention else ([], [])
    )

    patch_indices = []
    if use_attention:
        patch_size = model.visual.patch_embed.patch_size[0]
        patch_indices = _get_patch_indices(layout, patch_size)

    # Run and save chunks
    try:
        with tqdm(
            total=len(design_matrix),
            desc=experiment["experiment_id"],
            unit="trial",
            position=1,
            leave=False,
        ) as progress:

            for chunk_start in range(0, len(design_matrix), CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, len(design_matrix))
                chunk_path = chunk_dir / f"chunk_{chunk_start // CHUNK_SIZE:04d}.csv"

                if chunk_path.exists():
                    progress.update(chunk_end - chunk_start)
                    continue

                chunk_results = []

                for start in range(chunk_start, chunk_end, batch_size):
                    batch_df = design_matrix.iloc[
                        start:min(start + batch_size, chunk_end)
                    ]

                    images = [
                        torch.stack([
                            image_cache[image_id] for image_id in batch_df[col]
                        ]).to(device).float() / 255
                        for col in constituent_cols
                    ]

                    composites = make_composite(images, layout)
                    composites = (composites - mean) / std

                    attention_cache.clear()
                    with torch.no_grad():
                        composite_embeddings = model.encode_image(composites)

                    attentions = (
                        torch.stack(attention_cache, dim=1)
                        if use_attention else None
                    )

                    chunk_results.extend(compute_metrics(
                        batch_df,
                        composite_embeddings,
                        attentions,
                        metrics,
                        constituent_embeddings,
                        patch_indices,
                        constituent_cols,
                    ))

                    attention_cache.clear()
                    del attentions, composite_embeddings, composites, images

                    progress.update(len(batch_df))

                chunk_df = pd.concat(chunk_results, ignore_index=True)
                temp_path = chunk_path.with_suffix(".tmp")
                chunk_df.to_csv(temp_path, index=False)
                temp_path.replace(chunk_path)

    finally:
        for handle in handles:
            handle.remove()

    # Merge chunks
    temp_path = result_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for i, path in enumerate(sorted(chunk_dir.glob("chunk_*.csv"))):
            pd.read_csv(path).to_csv(f, index=False, header=(i == 0))

    temp_path.replace(result_path)
    print(f"Results saved: {result_path}")
    return result_dir


# %% Test
if __name__ == "__main__":
    from config import sweep_experiments

    experiment = next(sweep_experiments("main_sweep"))
    run_experiment(experiment)
