"""
Generate the design matrix for a given imageset, including trial-by-trial composition.
"""

from config.constants import N_TRIALS, N_PER_TRIAL, IMAGE_EXTENSIONS
from script.utils import traverse_imageset

import argparse
import random
from pathlib import Path
import pandas as pd
from math import perm


def get_image_ids(imageset_id: str, subset: str) -> list:
    """
    Get the image IDs for a given imageset and subset (if specified).
    """
    ratings_path = Path("stimuli", imageset_id, "ratings.csv")
    print(ratings_path)
    
    ratings_df = pd.read_csv(ratings_path)
    if subset:
        image_ids = ratings_df.loc[ratings_df["Category"] == subset, "Image_id"].tolist()
    else:
        image_ids = ratings_df["Image_id"].tolist()
    return image_ids

def generate_design_matrix(image_ids: list, n_trials: int, n_per_trial: int) -> pd.DataFrame:
    """
    Generate the design matrix for a given list of image IDs.
    """
    # check number of possible trials
    max_trials = perm(len(image_ids), n_per_trial)

    if n_trials > max_trials:
        generate_trials = max_trials
        print(
            f"Warning: requested {n_trials} trials, "
            f"but only {max_trials} unique trials are possible."
        )
    else:
        generate_trials = n_trials

    # sample, making sure unique trials
    random.seed(42)

    design_matrix = set()
    while len(design_matrix) < generate_trials:
        trial = tuple(random.sample(image_ids, n_per_trial))
        design_matrix.add(trial)

    design_matrix = sorted(design_matrix) # sort for consistent output

    # convert to DataFrame
    design_matrix_df = pd.DataFrame(
        design_matrix,
        columns=[f"Image_{i}" for i in range(n_per_trial)]
    )

    design_matrix_df.index.name = "Trial_id"
    design_matrix_df = design_matrix_df.reset_index()

    return design_matrix_df

def main():
    parser = argparse.ArgumentParser(description="Generate the design matrix for a given imageset (and subset).")
    parser.add_argument("imageset_id", help = "Imageset name (e.g. EmoMadrid, OASIS)")
    parser.add_argument("subset", nargs="?", default = None, help = "Subset name (e.g. People, Animal)")
    args = parser.parse_args() 
    imageset_id = args.imageset_id
    subset = args.subset

    image_ids = get_image_ids(imageset_id, subset)

    design_matrix = generate_design_matrix(image_ids, N_TRIALS, N_PER_TRIAL)
    print(design_matrix)

    # save to CSV
    #df = pd.DataFrame(list(design_matrix), columns=[f"Image_{i}" for i in range(N_PER_TRIAL)])
    #df.to_csv(f"design_matrix_{imageset_id}_{subset}.csv", index=False)

if __name__ == "__main__":
    main()

