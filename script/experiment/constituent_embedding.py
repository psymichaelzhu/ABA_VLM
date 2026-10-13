
"""
Generate and cache constituent embeddings for experiments.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from config import PROJECT_ROOT, SLIP_NORM_MEAN, SLIP_NORM_STD
from script.experiment.design_matrix import get_image_ids


# %% Item placement
def place_item(
    canvas: torch.Tensor,
    images: torch.Tensor,
    layout: dict,
    position: int,
) -> torch.Tensor:
    """Resize and place a batch of items on canvases."""
    w, h = layout["item_size"]
    x, y = layout["positions"][position]

    items = F.interpolate(
        images,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    canvas[:, :, y:y+h, x:x+w] = items
    return canvas


# %% Constituent construction
def make_constituent(
    images: torch.Tensor,
    layout: dict,
    present: str,
    position: int,
) -> torch.Tensor:
    """Construct a batch of constituent images."""
    if present == "isolate":
        return images

    w, h = layout["canvas_size"]
    background = torch.tensor(
        layout["background"],
        dtype=images.dtype,
        device=images.device,
    ) / 255

    canvas = background.view(1, 3, 1, 1).expand(
        len(images), 3, h, w
    ).clone()

    return place_item(canvas, images, layout, position)


# %% Composite construction
def make_composite(
    images: list[torch.Tensor],
    layout: dict,
) -> torch.Tensor:
    """Construct a batch of composite images."""
    canvas = make_constituent(images[0], layout, "semi", 0)

    for position, image in enumerate(images[1:], start=1):
        place_item(canvas, image, layout, position)

    return canvas


# %% Extraction
def extract_constituent_embeddings(
    imageset: str,
    layout: dict,
    embedding_config: dict,
    model,
    device,
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Extract embeddings for all images and positions."""
    image_ids = get_image_ids(imageset, "all")
    image_dir = PROJECT_ROOT / "stimuli" / imageset / "images"

    present = embedding_config["present"]
    encoder = getattr(model, embedding_config["embedding_function"])

    n_positions = len(layout["positions"])
    positions = range(n_positions) if present == "semi" else range(1)

    mean = torch.tensor(SLIP_NORM_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(SLIP_NORM_STD, device=device).view(1, 3, 1, 1)

    embeddings = {}

    with torch.no_grad(), tqdm(
        total=len(image_ids),
        desc=f"Constituent embeddings: {imageset}",
        unit="image",
    ) as progress:
        for start in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[start:start + batch_size]

            images = torch.stack([
                pil_to_tensor(
                    Image.open(image_dir / f"{image_id}.jpg").convert("RGB")
                ).float() / 255
                for image_id in batch_ids
            ]).to(device)

            for position in positions:
                batch = make_constituent(images, layout, present, position)
                features = encoder((batch - mean) / std).cpu()

                for image_id, embedding in zip(batch_ids, features):
                    if present == "isolate":
                        embeddings[image_id] = embedding.repeat(n_positions, 1)
                    else:
                        if position == 0:
                            embeddings[image_id] = torch.empty(
                                n_positions, embedding.numel(),
                                dtype=embedding.dtype,
                            )
                        embeddings[image_id][position] = embedding

            progress.update(len(batch_ids))

    return embeddings


# %% Cache
def ensure_constituent_embeddings(
    experiment: dict,
    model,
    device,
    overwrite: bool = False,
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Load or generate constituent embeddings for an experiment."""
    imageset = experiment["imageset"]
    model_id = experiment["model"]
    layout_id = experiment["layout_id"]
    layout = experiment["layout"]
    embedding_config = experiment["metrics"]["embedding_similarity"]

    filename = (
        f"{layout_id}__{embedding_config['present']}__"
        f"{embedding_config['embedding_function']}.pt"
    )
    path = (
        PROJECT_ROOT / "data" / "constituent_embeddings"
        / imageset / model_id / filename
    )

    if path.exists() and not overwrite:
        print(f"Loading constituent embeddings: {path}")
        return torch.load(path, map_location="cpu")

    embeddings = extract_constituent_embeddings(
        imageset, layout, embedding_config, model, device, batch_size
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)
    print(f"Generated constituent embeddings: {path}")
    return embeddings


# %% Test
if __name__ == "__main__":
    from config import sweep_experiments

    experiment = next(sweep_experiments("main_sweep"))
    imageset = experiment["imageset"]
    layout = experiment["layout"]

    image_ids = get_image_ids(imageset, "all")[:4]
    image_dir = PROJECT_ROOT / "stimuli" / imageset / "images"

    images = torch.stack([
        pil_to_tensor(
            Image.open(image_dir / f"{image_id}.jpg").convert("RGB")
        ).float() / 255
        for image_id in image_ids
    ])

    output_dir = PROJECT_ROOT / "data" / "test_constituent"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test constituent modes
    for present in ("semi", "isolate"):
        positions = range(len(layout["positions"])) if present == "semi" else range(1)

        for position in positions:
            batch = make_constituent(images, layout, present, position)
            to_pil_image(batch[0]).save(
                output_dir / f"{present}_{position}.png"
            )

    # Test pairwise composite
    composite = make_composite(
        [images[0:1], images[1:2]], layout
    )
    to_pil_image(composite[0]).save(output_dir / "composite.png")

    # Test 2x2 grid composite
    grid_layout = {
        "canvas_size": [224, 224],
        "item_size": [96, 96],
        "positions": [
            [0, 0], [128, 0],
            [0, 128], [128, 128],
        ],
        "background": [128, 128, 128],
    }

    grid = make_composite(
        [images[i:i+1] for i in range(4)], grid_layout
    )
    to_pil_image(grid[0]).save(output_dir / "grid_2x2.png")

    print(f"Test images saved to {output_dir}")
