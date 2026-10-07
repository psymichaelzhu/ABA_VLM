from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from config.constants import LAYOUT, SLIP_NORM_MEAN, SLIP_NORM_STD

from script.utils import resize_image_cache

def _build_item_batch(
    images: torch.Tensor,
    present: str,
    position_idx: Optional[int] = None,
) -> torch.Tensor:
    """Build item inputs in isolate or semi presentation."""

    if present == "semi":
        canvas_w, canvas_h = LAYOUT["canvas_size"]
        item_w, item_h = LAYOUT["item_size"]
        x, y = LAYOUT["positions"][position_idx]

        background = torch.tensor(LAYOUT["background"], dtype=torch.float32) / 255.0
        batch = background.view(1, 3, 1, 1).expand(
            len(images), 3, canvas_h, canvas_w
        ).clone()

        batch[:, :, y:y + item_h, x:x + item_w] = images

    else:
        batch = images

    return normalize(batch, mean=SLIP_NORM_MEAN, std=SLIP_NORM_STD)

def _extract_embeddings(
    model: torch.nn.Module,
    image_cache: dict[str, torch.Tensor],
    batch_size: int,
    present: str,
    position_idx: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Extract embeddings for all images in the cache."""

    image_ids = list(image_cache)
    device = next(model.parameters()).device
    embeddings = {}

    for start in tqdm(range(0, len(image_ids), batch_size)):
        batch_ids = image_ids[start:start + batch_size]
        images = torch.stack([image_cache[image_id] for image_id in batch_ids])

        batch = _build_item_batch(images, present, position_idx).to(device)

        with torch.no_grad():
            batch_embeddings = model.encode_image(batch).cpu()

        embeddings.update(zip(batch_ids, batch_embeddings))

    return embeddings

def _resolve_path(
    imageset_id: str,
    model_id: str,
    present: str,
    layout_id: Optional[str],
) -> Path:
    """Resolve the item-embedding cache path."""

    base = Path("data", "item_embeddings", imageset_id, model_id)

    if present == "semi":
        return base / "semi" / f"{layout_id}.pt"
    else:
        return base / "isolate.pt"

def get_item_embeddings(
    model: torch.nn.Module,
    image_cache: dict[str, torch.Tensor],
    imageset_id: str,
    model_id: str,
    present: str,
    layout_id: Optional[str] = None,
    batch_size: int = 64,
) -> dict:
    """Load cached item embeddings or create them if needed."""

    path = _resolve_path(imageset_id, model_id, present, layout_id)

    if path.exists():
        print(f"Loading item embeddings: {path}")
        return torch.load(path, map_location="cpu")
    else:
        print(f"Creating item embeddings: {path}")

        if present == "semi":
            image_cache = resize_image_cache(image_cache)

            embeddings = {
                f"item{i}": _extract_embeddings(
                    model, image_cache, batch_size, present, position_idx=i
                )
                for i in range(len(LAYOUT["positions"]))
            }
        else:
            embeddings = _extract_embeddings(model, image_cache, batch_size, present)

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, path)

        return embeddings