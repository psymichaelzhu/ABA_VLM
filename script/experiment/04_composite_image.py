"""
Generate a composite image given two item images and a layout.
"""

from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from config.constants import LAYOUT


def make_composite(dm_name: str, layout: dict, trial_id: int) -> Image.Image:
    """Create a composite image."""
    with open(Path("config", "design_matrix.yml"), "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    imageset = metadata[dm_name]["imageset_id"]

    dm_path = Path("data", "design_matrix", dm_name).with_suffix(".csv")
    trial = pd.read_csv(dm_path).iloc[trial_id]

    image_dir = Path("stimuli", imageset, "images")
    images = [
        Image.open(image_dir / f"{trial['Image_0']}.jpg").convert("RGB"),
        Image.open(image_dir / f"{trial['Image_1']}.jpg").convert("RGB"),
    ]

    composite = Image.new(
        "RGB",
        layout["canvas_size"],
        layout["background"]
    )

    for image, position in zip(images, layout["positions"]):
        composite.paste(image, position)

    return composite


def main():
    composite_image = make_composite("dm_862b31c6",LAYOUT,2)
    composite_image.show()
    
    
if __name__ == "__main__":
    main()