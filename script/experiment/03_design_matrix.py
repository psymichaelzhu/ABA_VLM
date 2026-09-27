"""
Generate the design matrix for a given imageset, including trial-by-trial composition.
"""

from config.constants import N_TRIALS, N_PER_TRIAL, SEED
from script.utils import traverse_imageset, log_metadata

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
    
    rng = random.Random(SEED)

    design_matrix = set()
    while len(design_matrix) < generate_trials:
        trial = tuple(rng.sample(image_ids, n_per_trial))
        design_matrix.add(trial)
    design_matrix = sorted(design_matrix)

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
    
    dm_key = log_metadata(
        "design_matrix",
        imageset_id= imageset_id,
        subset = subset,
        n_trials = N_TRIALS,
        n_per_trial = N_PER_TRIAL,
        seed = SEED
    )
    output_path = Path("data", "design_matrix", dm_key).with_suffix(".csv")

    if output_path.exists():
        print(f"Design matrix already exists: {dm_key}")
    else: 
        image_ids = get_image_ids(imageset_id, subset)

        design_matrix = generate_design_matrix(image_ids, N_TRIALS, N_PER_TRIAL)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        design_matrix.to_csv(output_path, index=False)
        print(f"Design matrix saved to {output_path}")

if __name__ == "__main__":
    main()

