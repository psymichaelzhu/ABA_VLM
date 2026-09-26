IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

STANDARD_IMAGE_SIZE = (224, 224)



STANDARDIZE_PIPELINE = {
    "OASIS": ["resize"],
    "EmoMadrid": ["crop_black_borders", "resize"]
}



N_TRIALS = 3
N_PER_TRIAL = 2