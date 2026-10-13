"""
Run all experiments in the active sweep.
"""

import multiprocessing as mp
from tqdm import tqdm

from config import ACTIVE_SWEEP, sweep_experiments
from script.experiment.experiment import run_experiment


# %% Runner
def _run_experiment(experiment):
    run_experiment(experiment)


def main():
    """Run the active experiment sweep."""
    experiments = list(sweep_experiments(ACTIVE_SWEEP))

    for experiment in tqdm(
        experiments,
        desc=f"Sweep: {ACTIVE_SWEEP}",
        unit="experiment",
    ):
        process = mp.Process(
            target=_run_experiment,
            args=(experiment,),
        )
        process.start()
        process.join()

        if process.exitcode != 0:
            raise RuntimeError(
                f"Experiment failed: {experiment['experiment_id']}"
            )


# %% Main
if __name__ == "__main__":
    main()