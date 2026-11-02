"""Few-shot calibration: fine-tune the saved sim-pretrained 62x AE (final_62x/trial16/seed*/ES_best_weights.pt) on k original-fibre
experimental TMs (the first k of the training split; scored on the 20 validation TMs). Pretraining is NOT repeated: ES weights are loaded.
Note: the original FT runs also restored the end-of-pretraining Adam state; that state was not saved, so this script starts
AdamW fresh (lr = finetune_lr, weight_decay from the trial). Run k = 5 and 10 too, so the whole curve shares one recipe.
Checkpoint choice uses the validation TMs (as in the main study), so the validation errors are slightly optimistic, like the rest.

Usage: python evaluation/fewshot_finetune.py                        # k = 1 2 3 5 10, seeds 42-46
       python evaluation/fewshot_finetune.py --ks 2 --seeds 42 --save-weights   # writes FewShot_exp_N2.pt next to ES_best_weights.pt
Weights are read from <TM_MODEL_ROOT>/final_62x/trial16/seed*/ (default: the weights/ folder of this repository);
--save-weights writes FewShot_exp_N<k>.pt into the same seed folder.
Output: <TM_OUT_ROOT>/pca_baseline/fewshot_exp.csv  (seed, k, val_error; k = 0 is the sim-pretrained model)
PCA comparator for the same k:  python evaluation/pca_baseline.py --fit-n 1 2 3 5 10
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths
from autoencoder_functions import TM_Autoencoder, phase_aligned_loss_wholeTM_perSample

parser = argparse.ArgumentParser()
parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
parser.add_argument("--max-ft-epochs", type=int, default=5000, help="epoch cap (the main final runs used 1000; 300 cut several runs off while still improving)")
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--save-weights", action="store_true")
parser.add_argument("--fixed-epochs", type=int, nargs="+", default=[],
                    help="fixed schedule, NO validation-based stopping: train for max(E) epochs and score the model after each listed E "
                         "(e.g. 25 50 100 200). Results go to fewshot_<target>_fixed.csv; the val TMs are only used to score, never to choose")
args = parser.parse_args()

AE_DIR = paths.MODEL_ROOT / "final_62x" / "trial16"
OUT_DIR = paths.OUT_ROOT / "pca_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)

stack = paths.load_exp_stack()
split_dir, target = paths.EXP_SPLIT_DIR, "exp"
train_idx = np.load(split_dir / "train_split_idx.npy")
TM_val = torch.tensor(np.ascontiguousarray(stack[np.load(split_dir / "val_split_idx.npy")]), dtype=torch.cfloat, device=device)
print(f"target {target}: {len(TM_val)} val TMs, {len(train_idx)} train TMs available")


def forward_loss(model, X):
    B = X.shape[0]
    X_in = torch.stack([X.real, X.imag], dim=1).reshape(B, 2, 205, 205, 145).permute(0, 1, 4, 2, 3)
    pred, _ = model(F.pad(X_in, (0, 3, 0, 3)), show_shapes=False)
    pred = pred[:, :, :, :205, :205].permute(0, 1, 3, 4, 2).reshape(B, 2, 42025, 145)
    return phase_aligned_loss_wholeTM_perSample(torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()), X)[0]


@torch.no_grad()
def evaluate(model):
    model.eval()
    total = 0.0
    for s in range(0, len(TM_val), 4):
        X = TM_val[s:s + 4]
        total += forward_loss(model, X).item() * len(X)
    return total / len(TM_val)


csv_path = OUT_DIR / f"fewshot_{target}.csv"
rows = {}                                  # (seed, k) -> val error; an existing file is kept, so separate runs (e.g. --ks 163) add to it
meta = {}                                  # (seed, k) -> (best_epoch, epochs_run); blank for rows from older runs
if csv_path.exists():
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["seed"]), int(r["k"]))
            rows[key] = float(r["val_error"])
            meta[key] = (r.get("best_epoch", ""), r.get("epochs_run", ""))
fixed_path = OUT_DIR / f"fewshot_{target}_fixed.csv"
fixed_rows = {}                            # (seed, k, epochs) -> val error, fixed-schedule mode
if args.fixed_epochs and fixed_path.exists():
    with open(fixed_path, newline="") as f:
        fixed_rows = {(int(r["seed"]), int(r["k"]), int(r["epochs"])): float(r["val_error"]) for r in csv.DictReader(f)}
for seed in args.seeds:
    d = AE_DIR / f"seed{seed}"
    hp = json.loads((d / "hyperparameters.json").read_text())["params"]
    es_state = torch.load(d / "ES_best_weights.pt", map_location="cpu")

    def fresh():
        m = TM_Autoencoder(enc_channels=[hp["enc1"], 2 * hp["enc1"], 3 * hp["enc1"]], latent_dim=8,
                           attn_heads=hp["attn_heads"], linformer_k=hp["linformer_k"]).to(device)
        m.load_state_dict(es_state)
        return m

    m0 = fresh()
    rows[(seed, 0)] = evaluate(m0)
    print(f"seed {seed} k=0 (sim-pretrained): {100 * rows[(seed, 0)]:.2f}%")
    del m0
    for k in args.ks:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
        rng = np.random.default_rng(seed)
        TM_ft = torch.tensor(np.ascontiguousarray(stack[train_idx[:k]]), dtype=torch.cfloat)    # nested prefixes of the train list
        model = fresh()
        opt = torch.optim.AdamW(model.parameters(), lr=hp["finetune_lr"], weight_decay=hp["weight_decay"])
        if args.fixed_epochs:                      # fixed schedule: no stopping rule, no checkpoint choice; score at the listed epoch counts
            checkpoints = set(args.fixed_epochs)
            for epoch in range(1, max(checkpoints) + 1):
                model.train()
                perm = rng.permutation(len(TM_ft))
                for s in range(0, len(perm), args.batch_size):
                    opt.zero_grad(set_to_none=True)
                    forward_loss(model, TM_ft[perm[s:s + args.batch_size]].to(device)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                if epoch in checkpoints:
                    fixed_rows[(seed, k, epoch)] = evaluate(model)
                    print(f"seed {seed} k={k} fixed {epoch} epochs: {100 * fixed_rows[(seed, k, epoch)]:.2f}%")
            del model, opt, TM_ft
            torch.cuda.empty_cache()
            continue
        best, best_state, bad, t0, best_epoch = float("inf"), None, 0, time.time(), -1
        for epoch in range(args.max_ft_epochs):
            model.train()
            perm = rng.permutation(len(TM_ft))
            for s in range(0, len(perm), args.batch_size):
                opt.zero_grad(set_to_none=True)
                forward_loss(model, TM_ft[perm[s:s + args.batch_size]].to(device)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            val = evaluate(model)
            if val < best:
                best, bad, best_epoch = val, 0, epoch
                best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()} if args.save_weights else None
            else:
                bad += 1
                if bad >= args.patience:
                    break
        rows[(seed, k)] = best
        meta[(seed, k)] = (best_epoch, epoch + 1)
        capped = epoch + 1 >= args.max_ft_epochs
        print(f"seed {seed} k={k}: {100 * best:.2f}%  (best epoch {best_epoch}, {epoch + 1} epochs, {time.time() - t0:.0f}s)"
              + ("  WARNING: hit the epoch cap, still improving" if capped else ""))
        if args.save_weights:
            torch.save(best_state, d / f"FewShot_{target}_N{k}.pt")
        del model, opt, TM_ft
        torch.cuda.empty_cache()
    if args.fixed_epochs:
        with open(fixed_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target", "seed", "k", "epochs", "val_error"])
            w.writerows([(target, s, k_, e_, round(v, 5)) for (s, k_, e_), v in sorted(fixed_rows.items())])
        continue
    with open(csv_path, "w", newline="") as f:                       # rewritten after every seed
        w = csv.writer(f)
        w.writerow(["target", "seed", "k", "val_error", "best_epoch", "epochs_run"])
        w.writerows([(target, s, k_, round(e, 5), *meta.get((s, k_), ("", ""))) for (s, k_), e in sorted(rows.items())])

if args.fixed_epochs:
    print("\nFixed-schedule summary (mean ± SD over seeds, % val error):")
    for k in args.ks:
        for e in sorted(set(args.fixed_epochs)):
            v = np.array([x for (s, k_, e_), x in fixed_rows.items() if k_ == k and e_ == e]) * 100
            if len(v):
                print(f"  k={k:3d}, {e:4d} epochs: {v.mean():6.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:5.2f}  (n={len(v)})")
    print("saved", fixed_path)
    raise SystemExit

print("\nSummary (mean ± SD over seeds, % val error):")
for k in sorted({0, *(k_ for _, k_ in rows)}):
    v = np.array([e for (s, k_), e in rows.items() if k_ == k]) * 100
    print(f"  k={k:3d}: {v.mean():6.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:5.2f}  (n={len(v)})")
old_runs = sorted(key for key, (_, n_ep) in meta.items() if str(n_ep) in ("", "300") and key[1] > 0)
if old_runs:
    print(f"\nNote: {len(old_runs)} stored runs have no epoch record or stopped at exactly 300 epochs (the old cap); rerun those (seed, k) "
          f"with the new cap if they matter: {old_runs[:12]}{' ...' if len(old_runs) > 12 else ''}")
print("saved", csv_path)
