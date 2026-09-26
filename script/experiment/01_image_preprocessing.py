"""
Preprocess the images in the Affective Image set.
"""

from config.constants import IMAGE_EXTENSIONS, STANDARD_IMAGE_SIZE, STANDARDIZE_PIPELINE
from script.utils import crop_black_borders, resize_image, traverse_imageset

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Preprocess the images in the Affective Image set.")
    parser.add_argument("imageset_id", help= "Imageset Name (e.g. EmoMadrid, OASIS)")
    args = parser.parse_args() 
    imageset_id = args.imageset_id


    image_paths = traverse_imageset(imageset_id, IMAGE_EXTENSIONS)

    output_dir = Path("stimuli",imageset_id,"images")
    output_dir.mkdir(exist_ok = True)

    for path in tqdm(image_paths, desc = "Processing images"):
        new_path = Path(output_dir,path.stem).with_suffix(".jpg")
        img = Image.open(path)
        if "crop_black_borders" in STANDARDIZE_PIPELINE[imageset_id]:
            img = crop_black_borders(img)
        if "resize" in STANDARDIZE_PIPELINE[imageset_id]:
            img = resize_image(img, STANDARD_IMAGE_SIZE)
        img = img.convert("RGB")
        img.save(new_path, "JPEG", quality=95)

if __name__ == "__main__":
    main()