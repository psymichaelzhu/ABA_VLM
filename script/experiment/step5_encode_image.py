# %% imports

import sys

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, normalize
import torch.nn.functional as F

from typing import Optional
from tqdm import tqdm


# Make SLIP importable
sys.path.insert(0, "external/SLIP")
import models

from config.constants import LAYOUT, SLIP_NORM_MEAN, SLIP_NORM_STD
from script.utils import get_device, get_image_ids, resize_image_cache
from script.experiment.step4_item_embedding import get_item_embeddings


# %% helpers

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load a pretrained SLIP-family model from checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = OrderedDict(
        (key.replace("module.", ""), value)
        for key, value in checkpoint["state_dict"].items()
    )

    args = checkpoint["args"]

    model = getattr(models, args.model)(
        rand_embed=False,
        ssl_mlp_dim=args.ssl_mlp_dim,
        ssl_emb_dim=args.ssl_emb_dim,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    return model, checkpoint

def register_attention_hooks(
    model: torch.nn.Module,
) -> tuple[list, list]:
    """
    Register forward hooks on all ViT attention dropout modules.
    """

    attention_cache = []
    handles = []

    def hook_fn(module, inputs, output):
        attention_cache.append(output.detach())

    for block in model.visual.blocks:
        handles.append(
            block.attn.attn_drop.register_forward_hook(hook_fn)
        )

    return attention_cache, handles


def load_image_cache(
    imageset_id: str,
    subset: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Load all images in the specified imageset (and subset) into a cache."""

    image_ids = get_image_ids(imageset_id, subset)

    image_dir = Path("stimuli", imageset_id, "images")

    cache = {}

    for image_id in image_ids:
        # all images have been standardized to 224x224 RGB jpg
        path = image_dir / f"{image_id}.jpg"

        with Image.open(path) as image:
            tensor = pil_to_tensor(image).float() / 255.0

        cache[image_id] = tensor

    return cache

def build_composite_batch(
    batch_design_matrix: pd.DataFrame,
    item_cache: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Build composites as tensors."""

    batch_size = len(batch_design_matrix)

    width, height = LAYOUT["canvas_size"]
    background = torch.tensor(
        LAYOUT["background"],
        dtype=torch.float32,
    ) / 255.0

    # [B, 3, H, W]
    tensor_batch = background.view(1, 3, 1, 1).expand(
        batch_size, 3, height, width
    ).clone()

    for i, row in enumerate(batch_design_matrix.itertuples(index=False)):

        for image_id, position in zip(
            [row.Image_0, row.Image_1],
            LAYOUT["positions"],
        ):
            image = item_cache[image_id]

            x, y = position
            _, h, w = image.shape

            tensor_batch[i, :, y:y + h, x:x + w] = image

    # SLIP official normalization
    tensor_batch = normalize(
        tensor_batch,
        mean=SLIP_NORM_MEAN,
        std=SLIP_NORM_STD,
    )

    return tensor_batch


# todo: remove to another debug file
def debug_show_composite_batch(batch_tensor: torch.Tensor):
    """Show composite images for debugging."""

    mean = torch.tensor(SLIP_NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(SLIP_NORM_STD).view(3, 1, 1)

    for i in range(min(5, len(batch_tensor))):
        image = batch_tensor[i].detach().cpu()
        image = torch.clamp(image * std + mean, 0, 1)
        image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        Image.fromarray(image).show()


def extract_batch(
    model: torch.nn.Module,
    image_batch: torch.Tensor,
    attention_cache: list,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings and attentions for the batch."""

    attention_cache.clear()

    image_batch = image_batch.to(device)

    with torch.no_grad():
        embeddings = model.encode_image(image_batch)

    attentions = torch.stack(attention_cache, dim=1)

    # Embedding shape: [batch, dimension]
    # Attention shape: [batch, layers, heads, tokens, tokens]
    return embeddings, attentions


# metric extraction

def embedding_similarity(
    composite_embeddings: torch.Tensor,
    item0_embeddings: torch.Tensor,
    item1_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute composite-item cosine similarity for each trial.
    """

    item0_similarity = F.cosine_similarity(
        composite_embeddings,
        item0_embeddings,
        dim=1,
    )

    item1_similarity = F.cosine_similarity(
        composite_embeddings,
        item1_embeddings,
        dim=1,
    )

    # shape: [batch]
    return item0_similarity, item1_similarity


def _aggregate_heads(
    attentions: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """
    Aggregate attention heads.
    """

    if method == "mean":
        return attentions.mean(dim=-3)

    if method == "max":
        return attentions.max(dim=-3).values

    if method == "min":
        return attentions.min(dim=-3).values

    raise ValueError(f"Unknown head aggregation: {method}")


def _discard_attention(
    attention: torch.Tensor,
    discard_ratio: float,
) -> torch.Tensor:
    """
    Set the lowest attention values to zero.
    """

    if discard_ratio > 0:
        flat = attention.flatten(start_dim=2)
        n_discard = int(flat.shape[-1] * discard_ratio)

        if n_discard > 0:
            indices = flat.argsort(dim=-1)[..., :n_discard]
            flat.scatter_(-1, indices, 0)
            attention = flat.reshape_as(attention)

    return attention


def _add_residual_and_normalize(
    attention: torch.Tensor,
) -> torch.Tensor:
    """
    Add identity residual connections and row-normalize attention.
    """

    n_tokens = attention.shape[-1]

    identity = torch.eye(
        n_tokens,
        device=attention.device,
        dtype=attention.dtype,
    ).view(1, 1, n_tokens, n_tokens)

    attention = attention + identity
    attention = attention / attention.sum(dim=-1, keepdim=True)

    return attention


def attention_rollout(
    attentions: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Compute attention rollout across transformer layers."""

    attention = _aggregate_heads(
        attentions,
        config["head_aggregation"],
    )

    attention = _discard_attention(
        attention,
        config["discard_ratio"],
    )

    attention = _add_residual_and_normalize(attention)

    rollout = attention[:, 0]

    for layer in range(1, attention.shape[1]):
        rollout = attention[:, layer] @ rollout

    # shape: [batch, tokens, tokens]
    return rollout


def attention_layer(
    attentions: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Extract attention matrix from one transformer layer."""

    attention = attentions[:, config["layer"]]

    attention = _aggregate_heads(
        attention,
        config["head_aggregation"],
    )

    # shape: [batch, tokens, tokens]
    return attention


def _region_summarize(
    region: torch.Tensor,
    summary: str,
) -> torch.Tensor:
    """Summarize attention values within a region."""

    if summary == "mean":
        return region.mean(dim=1)

    if summary == "max":
        return region.max(dim=1).values

    if summary == "min":
        return region.min(dim=1).values

    raise ValueError(f"Unknown region summary: {summary}")


def item_attention_summary(
    attention: torch.Tensor,
    summary: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Summarize CLS attention within each item region.
    """

    patch_attention = attention[:, 0, 1:]

    item0_region = patch_attention[:, LAYOUT["patch_indices"]["item0"]]
    item1_region = patch_attention[:, LAYOUT["patch_indices"]["item1"]]

    # shape: [batch]
    return (
        _region_summarize(item0_region, summary),
        _region_summarize(item1_region, summary),
    )

def build_metric_df(
    trial_ids,
    item0_values: torch.Tensor,
    item1_values: torch.Tensor,
    metric: str,
    config_key: str,
) -> pd.DataFrame:
    """Build a trial-level DataFrame for the metric values."""

    batch_size = len(trial_ids)

    return pd.DataFrame({
        "Trial_id": [trial_id for trial_id in trial_ids for _ in range(2)],
        "Item": [0, 1] * batch_size,
        "Metric": [metric] * (2 * batch_size),
        "Config_key": [config_key] * (2 * batch_size),
        "Value": torch.stack(
            [item0_values, item1_values],
            dim=1,
        ).reshape(-1).detach().cpu().numpy(),
    })

def main():
    device = get_device()

    checkpoint_id = "clip_base_25ep"
    designmatrix_id = "dm_0bdf71da"
    imageset_id = "EmoMadrid"
    layout_id = "layout_01"
    subset = None
    batch_size = 32

    rollout_config = {
        "head_aggregation": "max",
        "discard_ratio": 0.7,
        "region_summary": "mean",
    }

    layer_config = {
        "layer": -1,
        "head_aggregation": "max",
        "region_summary": "mean",
    }

    checkpoint_path = Path(f"checkpoints/fb_slip/{checkpoint_id}.pt")
    dm_path = Path(f"data/design_matrix/{designmatrix_id}.csv")

    # load model and data
    model, checkpoint = load_model(checkpoint_path, device)
    design_matrix = pd.read_csv(dm_path)

    image_cache = load_image_cache(imageset_id, subset)
    item_cache = resize_image_cache(image_cache)

    # load or create item embeddings before registering attention hooks
    item_embeddings = get_item_embeddings(
        model=model,
        image_cache=image_cache,
        imageset_id=imageset_id,
        model_id=checkpoint_id,
        present="semi",
        layout_id=layout_id,
        batch_size=batch_size,
    )

    # register attention hooks for composite extraction
    attention_cache, handles = register_attention_hooks(model)


    metric_dfs = []

    try:
        for start in tqdm(
            range(0, len(design_matrix), batch_size),
            desc="Running experiment",
            unit="batch",
        ):
            batch_df = design_matrix.iloc[start:start + batch_size]

            image_batch = build_composite_batch(batch_df, item_cache)

            composite_embeddings, attentions = extract_batch(
                model,
                image_batch,
                attention_cache,
                device,
            )

            # item embeddings corresponding to each trial
            item0_embeddings = torch.stack([
                item_embeddings["item0"][image_id]
                for image_id in batch_df["Image_0"]
            ]).to(device)

            item1_embeddings = torch.stack([
                item_embeddings["item1"][image_id]
                for image_id in batch_df["Image_1"]
            ]).to(device)

            trial_ids = batch_df["Trial_id"]

            # embedding similarity
            item0_values, item1_values = embedding_similarity(
                composite_embeddings,
                item0_embeddings,
                item1_embeddings,
            )

            metric_dfs.append(
                build_metric_df(
                    trial_ids,
                    item0_values,
                    item1_values,
                    metric="embedding_similarity",
                    config_key="semi",
                )
            )

            # attention rollout
            rollout = attention_rollout(attentions, rollout_config)

            item0_values, item1_values = item_attention_summary(
                rollout,
                rollout_config["region_summary"],
            )

            metric_dfs.append(
                build_metric_df(
                    trial_ids,
                    item0_values,
                    item1_values,
                    metric="attention_rollout",
                    config_key="max_0.7_mean",
                )
            )

            # last-layer attention
            layer_attention = attention_layer(attentions, layer_config)

            item0_values, item1_values = item_attention_summary(
                layer_attention,
                layer_config["region_summary"],
            )

            metric_dfs.append(
                build_metric_df(
                    trial_ids,
                    item0_values,
                    item1_values,
                    metric="attention_layer",
                    config_key="last_max_mean",
                )
            )

        results = pd.concat(metric_dfs, ignore_index=True)

        output_path = Path(
            "data",
            "results",
            f"{checkpoint_id}_{designmatrix_id}.csv",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

        print("\nExperiment complete.")
        print(f"Results saved to: {output_path}")
        print(f"Result rows: {len(results):,}")
        print(results.head())

    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()