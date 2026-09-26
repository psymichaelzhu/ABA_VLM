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
