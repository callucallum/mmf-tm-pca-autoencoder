"""Train/validation/test split of the simulated 1 m TM pool and the composite 5 m index files built from it.

Each composite 5 m TM is the product of five *distinct* 1 m TMs drawn from a single split, so no 1 m TM is shared between
training, validation and test composites. Training sets of 500, 1,000, 2,000 (and 3,000) composites are nested
prefixes of one array; validation and test sets have 250 composites each.

The index files are released in splits/indices/sim/. This script regenerates them (seed 42) and reports whether they match
the files already in --out-dir. Existing files are only replaced with --overwrite.

Usage: python splits/train_test_split_sim_indices_5m.py [--sim-dir DIR] [--out-dir DIR] [--overwrite]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import paths

parser = argparse.ArgumentParser()
parser.add_argument("--sim-dir", default=str(paths.SIM_DIR), help="folder with the simulated 1 m TMs (default: TM_SIM_DIR)")
parser.add_argument("--out-dir", default=str(paths.SIM_SPLIT_DIR), help="where to write the index files (default: TM_SIM_SPLIT_DIR)")
parser.add_argument("--overwrite", action="store_true", help="replace existing files that differ from the regenerated indices")
args = parser.parse_args()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


def save_idx(name, idx):
    f = out_dir / f"{name}.npy"
    if f.exists():
        same = np.array_equal(np.load(f), idx)
        print(f"{name}: shape {idx.shape}; existing file {'matches' if same else 'DIFFERS from'} the regenerated indices")
        if same or not args.overwrite:
            return
    np.save(f, idx)
    print(f"{name}: shape {idx.shape} written to {f}")


# The pool is the first 5,000 TM*.pkl files (TM0000 ... TM4999, with or without the -GPU suffix) in sorted order.
num_to_load = 5000
N = min(num_to_load, sum(1 for p in Path(args.sim_dir).iterdir() if p.name.startswith("TM") and p.name.endswith(".pkl")))
indices = np.arange(N)
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

print(f'Train: {len(train_idx)}, first 10: {train_idx[0:10]}')
print(f'Val: {len(val_idx)}, first 10: {val_idx[0:10]}')
print(f'Test: {len(test_idx)}, first 10: {test_idx[0:10]}')

# --- composite 5m index generation, per split, no cross-split leakage ---
N_SEGMENTS = 5
rng = np.random.default_rng(seed=42)  # fixed for reproducibility

def build_composite_indices(base_idx_pool, n_composite, n_segments, rng):
    """
    Draw n_composite rows of n_segments UNIQUE indices each,
    sampled only from base_idx_pool (a single split's indices).
    """
    composite = np.empty((n_composite, n_segments), dtype=base_idx_pool.dtype)
    for i in range(n_composite):
        composite[i] = rng.choice(base_idx_pool, size=n_segments, replace=False)
    return composite

# generate at max size per split, take prefixes for smaller N to preserve nesting
N_COMPOSITE_TRAIN_MAX = 2000  # largest training set size in the original sweep
N_COMPOSITE_VAL = 250
N_COMPOSITE_TEST = 250

train_composite_idx = build_composite_indices(train_idx, N_COMPOSITE_TRAIN_MAX, N_SEGMENTS, rng)
val_composite_idx = build_composite_indices(val_idx, N_COMPOSITE_VAL, N_SEGMENTS, rng)
test_composite_idx = build_composite_indices(test_idx, N_COMPOSITE_TEST, N_SEGMENTS, rng)

save_idx("train_composite5m_idx_2000", train_composite_idx)
save_idx("val_composite5m_idx", val_composite_idx)
save_idx("test_composite5m_idx", test_composite_idx)
save_idx("train_composite5m_idx_500", train_composite_idx[:500])
save_idx("train_composite5m_idx_1000", train_composite_idx[:1000])

# --- extend the 2000 set to 3000, guaranteeing the 2000 set is a strict prefix ---
# Uses a separate RNG stream, so this can never perturb train_idx/val_idx/test_idx or the existing val/test composite sets
# used to evaluate the 500/1000/2000 studies.
N_COMPOSITE_TRAIN_3000 = 3000
n_new_rows = N_COMPOSITE_TRAIN_3000 - len(train_composite_idx)
extend_rng = np.random.default_rng(np.random.SeedSequence(42).spawn(1)[0])  # independent stream derived from seed 42, won't replay `rng`'s draws
new_composite_rows = build_composite_indices(train_idx, n_new_rows, N_SEGMENTS, extend_rng)
train_composite_idx_3000 = np.concatenate([train_composite_idx, new_composite_rows], axis=0)

assert np.array_equal(train_composite_idx_3000[:2000], train_composite_idx), \
    "3000 set must contain the existing 2000 set as an exact prefix"

assert not np.array_equal(new_composite_rows[0], train_composite_idx[0]), \
    "extend_rng produced the same first row as the original 2000 set - it replayed rng's stream instead of an independent one"

save_idx("train_composite5m_idx_3000", train_composite_idx_3000)
