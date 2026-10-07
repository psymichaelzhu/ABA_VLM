from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# -----------------
# helpers
# -----------------

def _resolve_layout(
    layout: dict,
    model_config: dict,
) -> dict:
    """Validate layout and add flattened patch indices."""

    patch_size = model_config["patch_size"]

    canvas_w, canvas_h = layout["canvas_size"]
    item_w, item_h = layout["item_size"]

    if canvas_w % patch_size != 0 or canvas_h % patch_size != 0:
        raise ValueError("Canvas size must be divisible by patch size.")

    if item_w % patch_size != 0 or item_h % patch_size != 0:
        raise ValueError("Item size must be divisible by patch size.")

    grid_w = canvas_w // patch_size

    patch_indices = {}

    for i, (x, y) in enumerate(layout["positions"]):

        if x % patch_size != 0 or y % patch_size != 0:
            raise ValueError(
                f"Item {i} position {(x, y)} is not patch-aligned."
            )

        if x + item_w > canvas_w or y + item_h > canvas_h:
            raise ValueError(
                f"Item {i} exceeds canvas boundary."
            )

        x0 = x // patch_size
        y0 = y // patch_size
        w = item_w // patch_size
        h = item_h // patch_size

        patch_indices[f"item{i}"] = [
            row * grid_w + col
            for row in range(y0, y0 + h)
            for col in range(x0, x0 + w)
        ]

    layout["patch_indices"] = patch_indices

    return layout



# --------------------
# Image configuration
# --------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"
}

STANDARD_IMAGE_SIZE = (224, 224)

SLIP_NORM_MEAN = [0.485, 0.456, 0.406]
SLIP_NORM_STD = [0.229, 0.224, 0.225]

STANDARDIZE_PIPELINE = {
    "OASIS": ["resize"],
    "EmoMadrid": ["crop_black_borders", "resize"],
}


# --------------------
# Experiment configuration
# --------------------

N_TRIALS = 20000
N_PER_TRIAL = 2
SEED = 42


# --------------------
# Model configuration
# --------------------

MODEL_CONFIG = {
    "patch_size": 16,
}

# --------------------
# Layout configuration
# --------------------
LAYOUT = {
    "canvas_size": (224, 224),
    "positions": [
        (0, 64),
        (128, 64),
    ],
    "background": (128, 128, 128),
    "item_size": (96, 96),
}

LAYOUT = _resolve_layout(
    LAYOUT,
    MODEL_CONFIG,
)

# --------------------
# Metric configuration
# --------------------
METRIC_CONFIGS = {
    "visual": {
        "embedding_type": "visual",
    },
    "max_d07_mean": {
        "head_aggregation": "max",
        "discard_ratio": 0.7,
        "region_summary": "mean",
    },
}

# --------------------
# Debug
# --------------------
#todo: remove to another file
def _debug_layout():
    """Visualize layout and verify patch indices."""

    patch_size = MODEL_CONFIG["patch_size"]

    canvas_w, canvas_h = LAYOUT["canvas_size"]
    item_w, item_h = LAYOUT["item_size"]

    grid_w = canvas_w // patch_size
    grid_h = canvas_h // patch_size

    image = Image.new(
        "L",
        LAYOUT["canvas_size"],
        color=0,
    )

    draw = ImageDraw.Draw(image)

    colors = [255, 128]

    for i, ((x, y), color) in enumerate(
        zip(LAYOUT["positions"], colors)
    ):
        draw.rectangle(
            [
                x,
                y,
                x + item_w - 1,
                y + item_h - 1,
            ],
            fill=color,
        )

        # Verify resolved patch indices
        for index in LAYOUT["patch_indices"][f"item{i}"]:
            row = index // grid_w
            col = index % grid_w

            center_x = col * patch_size + patch_size // 2
            center_y = row * patch_size + patch_size // 2

            assert image.getpixel(
                (center_x, center_y)
            ) == color

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)

    for x in range(0, canvas_w + 1, patch_size):
        ax.axvline(x - 0.5, color="red", linewidth=1)

    for y in range(0, canvas_h + 1, patch_size):
        ax.axhline(y - 0.5, color="red", linewidth=1)

    for row in range(grid_h):
        for col in range(grid_w):
            index = row * grid_w + col

            ax.text(
                col * patch_size + patch_size / 2,
                row * patch_size + patch_size / 2,
                str(index),
                color="red",
                fontsize=8,
                ha="center",
                va="center",
            )

    print("Item 0:", LAYOUT["patch_indices"]["item0"])
    print("Item 1:", LAYOUT["patch_indices"]["item1"])
    print("Layout test passed.")

    ax.set_title("Layout visualization with patch indices")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


if __name__ == "__main__":
    _debug_layout()