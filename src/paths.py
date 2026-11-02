"""Shared path configuration for every script in this repository.

Each location can be overridden with an environment variable. Environment variables are used (rather than command-line
flags) because the sub-batch training scripts already take positional arguments from their run_* drivers.

    TM_EXP_DATA       experimental TM stack, key 'TM', shape (204, 42025, 145)  [processed_dataset.npz from the
                      mmf-stepindex-tms-optimlab dataset repo; default ../mmf-stepindex-tms-optimlab/processed_dataset.npz]
    TM_SIM_DIR        simulated 1 m TMs (TM0000.pkl ... TM4999.pkl or TMxxxx-GPU.pkl), dist_pim_to_real.pkl and
                      prox_real_to_pim.pkl                                          [default ../mmf-stepindex-tms-optimlab/Simulated_(TMM)]
    TM_INPUTS         orthonormal_random_inputs_371x371_145.npy (the 145 orthonormal input patterns, released with the simulated TMs)
                                                                                    [default <TM_SIM_DIR>/orthonormal_random_inputs_371x371_145.npy]
    TM_SIM_SPLIT_DIR  train/val/test composite-5 m index files                      [default <repo>/splits/indices/sim]
    TM_EXP_SPLIT_DIR  train/val/test split indices of the 204 experimental TMs      [default <repo>/splits/indices/exp]
    TM_OUT_ROOT       where training scripts write checkpoints, logs, csv files     [default <repo>/outputs]
    TM_MODEL_ROOT     where evaluation scripts read trained checkpoints from        [default <repo>/weights]
                      (folder holding final_62x/trial16/seed42/..., exp_only_62x/...; set it to
                      TM_OUT_ROOT to evaluate checkpoints you trained yourself)
    TM_OPTUNA_DB      Optuna study (sqlite) holding the 27 trials of the final-configuration search; Final_62x_ES_FT.py reads the chosen
                      trial from it, Optuna_62x_ES_FT.py creates/resumes it  [default <repo>/optuna/TMM_62x_ES_FT_Optuna.db]
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(name, default):
    return Path(os.environ.get(name, str(default))).expanduser()


EXP_DATA = _path("TM_EXP_DATA", REPO_ROOT.parent / "mmf-stepindex-tms-optimlab" / "processed_dataset.npz")
SIM_DIR = _path("TM_SIM_DIR", REPO_ROOT.parent / "mmf-stepindex-tms-optimlab" / "Simulated_(TMM)")
INPUTS_FILE = _path("TM_INPUTS", SIM_DIR / "orthonormal_random_inputs_371x371_145.npy")
SIM_SPLIT_DIR = _path("TM_SIM_SPLIT_DIR", REPO_ROOT / "splits" / "indices" / "sim")
EXP_SPLIT_DIR = _path("TM_EXP_SPLIT_DIR", REPO_ROOT / "splits" / "indices" / "exp")
OUT_ROOT = _path("TM_OUT_ROOT", REPO_ROOT / "outputs")
MODEL_ROOT = _path("TM_MODEL_ROOT", REPO_ROOT / "weights")
OPTUNA_DB = _path("TM_OPTUNA_DB", REPO_ROOT / "optuna" / "TMM_62x_ES_FT_Optuna.db")


def load_exp_stack():
    """Memory-mapped experimental TM stack, shape (204, 42025, 145)."""
    import numpy as np
    return np.load(EXP_DATA, allow_pickle=True, mmap_mode="r")["TM"]
