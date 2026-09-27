import sys
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


# Make SLIP importable
sys.path.insert(0, "external/SLIP")
import models

from script.utils import get_device

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load a pretrained SLIP-family model from checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        state_dict[key.replace("module.", "")] = value

    old_args = checkpoint["args"]

    model = getattr(models, old_args.model)(
        rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim,
        ssl_emb_dim=old_args.ssl_emb_dim,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    return model, checkpoint


def get_preprocess():
    """Return the official SLIP evaluation preprocessing."""
    return transforms.Compose([
        # resize, crop and convert are ignored since input has been standardized to 224x224 and RGB
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def register_attention_hooks(model: torch.nn.Module) -> tuple[list, list]:
    """
    Register forward hooks on all ViT attention dropout modules.

    Returns:
        attention_cache: [n_images, n_layers, n_heads, n_tokens, n_tokens]
        handles: list of hook handles, for later removal.
    """
    attention_cache = []
    handles = []

    def hook_fn(module, inputs, output):
        attention_cache.append(output.detach())

    for block in model.visual.blocks:
        handle = block.attn.attn_drop.register_forward_hook(hook_fn)
        handles.append(handle)

    return attention_cache, handles

def remove_hooks(handles: list):
    """Remove all registered hooks."""
    for handle in handles:
        handle.remove()

def encode_images(model: torch.nn.Module, images: list[Image.Image], preprocess: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """
    Encode multiple images with the visual encoder.
    
    Returns:
        embeddings: [n_images, embedding_dim]
    """
    images = [preprocess(image) for image in images]
    image_batch = torch.stack(images, dim=0).to(device)

    with torch.no_grad():
        embeddings = model.encode_image(image_batch)

    return embeddings

def main(model_type: str):
    checkpoint_path = Path(
        f"checkpoints/fb_slip/{model_type}_base_25ep.pt"
    )

    image_paths = [
        Path("stimuli/OASIS/images/Acorns 1.jpg"),
        Path("stimuli/OASIS/images/Acorns 2.jpg"),
        Path("stimuli/OASIS/images/Cups 1.jpg"),
        Path("stimuli/OASIS/images/Cups 3.jpg"),
        ]

    images = [
        Image.open(path).convert("RGB")
        for path in image_paths
    ]

    device = get_device()

    # Load model
    model, checkpoint = load_model(checkpoint_path, device)
    print("Model:", checkpoint["args"].model)

    # Preprocessing
    preprocess = get_preprocess()

    # register attention hooks
    attention_cache, handles = register_attention_hooks(model)

    # Extract embeddings
    embeddings = encode_images(
        model,
        images,
        preprocess,
        device
    )
    print("Embedding shape:", embeddings.shape)
    # [n_images, embedding_dim]

    # Extract attention
    attentions = torch.stack(attention_cache, dim=1)
    print("Attention shape:", attentions.shape)
    # [n_images, n_layers, n_heads, n_tokens, n_tokens]

    # Remove attention hooks
    remove_hooks(handles)

    
if __name__ == "__main__":
    main("slip")
    main("clip")
    main("simclr")

