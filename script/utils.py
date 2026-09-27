###########################################################
# Image processing
###########################################################

from pathlib import Path
from PIL import Image

def crop_black_borders(img: Image.Image) -> Image.Image:
    """
    remove black borders
    """
    bbox = img.convert("L").getbbox()
    if bbox:
        return img.crop(bbox)
    return img

def resize_image(img: Image.Image, target_size: tuple) -> Image.Image:
    """
    resize image to target size
    """
    return img.resize(target_size, Image.Resampling.LANCZOS)


def traverse_imageset(imageset_id: str, image_extensions: set) -> list:
    """
    Traverse the imageset and collect all image paths."
    """
    image_paths = []

    imageset_root = Path("stimuli",imageset_id,"raw","images")
    

    for path in imageset_root.rglob("*"):
        if path.suffix.lower() in image_extensions:
            image_paths.append(path)

    print(len(image_paths))
    return image_paths


###########################################################
# Configuration I/O
###########################################################


import hashlib
from datetime import datetime
from pathlib import Path
import json
import yaml


def log_metadata(config_name: str, **kwargs) -> str:
    """"
    Generate the hash key for a unique configuration and log its meatadata to a YAML file.
    """
    config_path = Path("config", config_name).with_suffix(".yml")
    config_alias = "".join(word[0] for word in config_name.split("_"))

    param_json_str = json.dumps(kwargs, sort_keys=True)
    config_hash = hashlib.blake2b(param_json_str.encode("utf-8"),digest_size=4).hexdigest()
    unique_key = f"{config_alias}_{config_hash}"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            existing_metadata = yaml.safe_load(f) or {}
    else:
        existing_metadata = {}

    if unique_key in existing_metadata:
        print(f"Configuration already exists. Key: {unique_key}")
        return unique_key

    existing_metadata[unique_key] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        **kwargs
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing_metadata,f,sort_keys=False)

    print(f"New configuration logged, Key: {unique_key}")
    return unique_key







###########################################################
# test
###########################################################




def main():
    log_metadata("design_matrix",
                 n_trials = 3,
                 n_per_trial = 2,
                 imageset = "OASIS")
    
    
if __name__ == "__main__":
    main()
