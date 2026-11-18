"""Frozen split of the 204 experimental TMs: 163 train / 20 validation / 21 test (random_state=42), plus the nested
training subsets used for the fine-tuning data-scaling studies (first 5, 10, 25, 50, 100, 125, 150 training TMs).

The resulting index files are released in splits/indices/exp/. This script regenerates them from the number of TMs in
processed_dataset.npz (204) and reports whether they match the files already in --out-dir. The indices are drawn from
np.arange(204), so this confirms the split procedure and the TM count, not the row order. The row order is fixed by
processed_dataset_paths.txt in the mmf-tm-rm-experimental dataset repo (Experiment1/01 ... Experiment3/20, numerical
order), and the released indices refer to it. Existing files are only replaced with --overwrite.

Usage: python splits/train_test_split_exp_indices.py [--out-dir DIR] [--overwrite]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import paths

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=str(paths.EXP_SPLIT_DIR), help="where to write the index files (default: TM_EXP_SPLIT_DIR)")
parser.add_argument("--overwrite", action="store_true", help="replace existing files that differ from the regenerated indices")
args = parser.parse_args()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

TM_stack = paths.load_exp_stack()

N = len(TM_stack)
indices = np.arange(N)

train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

splits = {"train_split_idx": train_idx, "val_split_idx": val_idx, "test_split_idx": test_idx}
for n in (5, 10, 25, 50, 100, 125, 150):
    splits[f"train_split_idx_{n}"] = train_idx[:n]

for name, idx in splits.items():
    f = out_dir / f"{name}.npy"
    if f.exists():
        same = np.array_equal(np.load(f), idx)
        print(f"{name}: {len(idx)} indices; existing file {'matches' if same else 'DIFFERS from'} the regenerated indices")
        if same or not args.overwrite:
            continue
    np.save(f, idx)
    print(f"{name}: {len(idx)} indices written to {f}")
