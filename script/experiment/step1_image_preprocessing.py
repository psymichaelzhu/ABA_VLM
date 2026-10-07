"""
Preprocess the images in the Affective Image set.
"""

from config.constants import IMAGE_EXTENSIONS, STANDARD_IMAGE_SIZE, STANDARDIZE_PIPELINE

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse


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