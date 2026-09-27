IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

STANDARD_IMAGE_SIZE = (224, 224)



STANDARDIZE_PIPELINE = {
    "OASIS": ["resize"],
    "EmoMadrid": ["crop_black_borders", "resize"]
}



N_TRIALS = 3
N_PER_TRIAL = 2
SEED = 42



LAYOUT = {
    "canvas_size": (448, 224),
    "positions": [(0, 0), (224, 0)],
    "background": (255, 255, 255),
}