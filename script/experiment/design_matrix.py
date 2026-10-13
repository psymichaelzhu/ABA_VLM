
"""
Generate and cache design matrices for experiments.
"""

import random
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT


# %% Image IDs
def get_image_ids(imageset_id: str, subset: str) -> list[str]:
    """Get image IDs from a standardized imageset."""
    ratings_path = PROJECT_ROOT / "stimuli" / imageset_id / "ratings.csv"
    ratings = pd.read_csv(ratings_path)

    if subset != "all":
        ratings = ratings[ratings["Category"] == subset]

    return sorted(ratings["Image_id"].unique().tolist())


# %% Generation
def generate_design_matrix(
    image_ids: list[str],
    n_trials: int,
    n_items_per_trial: int,
    sampling_seed: int,
) -> pd.DataFrame:
    """Generate unique ordered trials in random order."""
    rng = random.Random(sampling_seed)
    trials = []
    seen = set()

    while len(trials) < n_trials:
        trial = tuple(rng.sample(image_ids, n_items_per_trial))
        if trial not in seen:
            seen.add(trial)
            trials.append(trial)

    columns = [f"Constituent_{i}" for i in range(n_items_per_trial)]
    df = pd.DataFrame(trials, columns=columns)
    df.insert(0, "Trial_id", range(len(df)))
    return df


# %% Cache
def ensure_design_matrix(
    experiment: dict,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Load or generate the design matrix for an experiment."""
    imageset = experiment["imageset"]
    subset = experiment["subset"]
    design = experiment["trial_design"]

    matrix_id = (
        f"{imageset}__{subset}__"
        f"{design['n_items_per_trial']}items__"
        f"{design['n_trials']}trials__seed{design['sampling_seed']}"
    )
    path = PROJECT_ROOT / "data" / "design_matrix" / f"{matrix_id}.csv"

    if path.exists() and not overwrite:
        print(f"Loading design matrix: {path}")
        return pd.read_csv(path), path

    df = generate_design_matrix(get_image_ids(imageset, subset), **design)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Generated design matrix: {path}")
    return df, path


# %% Test
if __name__ == "__main__":
    from config import sweep_experiments

    experiment = next(sweep_experiments("main_sweep"))
    df, path = ensure_design_matrix(experiment, overwrite=True)

    print(f"Experiment: {experiment['experiment_id']}")
    print(f"Design matrix: {path}")
    print(f"Shape: {df.shape}")
    print(df.head())
