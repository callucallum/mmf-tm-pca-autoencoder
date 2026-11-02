"""FINAL held-out evaluation of the original-fibre models. Run ONCE, after every model, table and figure is frozen.

Scores a FIXED list of models (below, printed before anything runs) on the held-out original-fibre test split (21 TMs).
Metric: phase-aligned relative Frobenius error (mean over TMs), the same as every validation number.
No checkpoint, hyper-parameter or model choice may change after seeing these numbers; if something must change, write down what and why.
A lock file (final_test_DONE.txt) stops a second run unless --allow-rerun is given (and the rerun is then reported as such).
All AE weights are seed 42 of trial 16 (final_62x); PCA fits are deterministic.
(The paper's test table also scores a second, collaborator fibre whose data are not public; those rows cannot be reproduced here.)

Usage: python evaluation/final_test_eval.py --plan        # print the model list and exit (check it, then freeze)
       python evaluation/final_test_eval.py --device cuda
Output: <TM_OUT_ROOT>/final_test_results.csv  (model, dataset, test_error, n_tm)
Weights are read from <TM_MODEL_ROOT>/final_62x/trial16/seed42/ (default: the weights/ folder of this repository).
"""
import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths
from autoencoder_functions import TM_Autoencoder, phase_aligned_loss_wholeTM_perSample

parser = argparse.ArgumentParser()
parser.add_argument("--plan", action="store_true", help="print the fixed model list and exit")
parser.add_argument("--allow-rerun", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

AE_DIR = paths.MODEL_ROOT / "final_62x" / "trial16" / f"seed{args.seed}"
paths.OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT, LOCK = paths.OUT_ROOT / "final_test_results.csv", paths.OUT_ROOT / "final_test_DONE.txt"
EXP_DIR = paths.EXP_SPLIT_DIR
PIXELS, N_IN, K62 = 42025, 145, 1352

# ── the frozen model list ─────────────────────────────────────────────────────
AE_MODELS = [("AE sim-pretrained", "ES_best_weights.pt"),
             ("AE specialist, original exp, 2 TMs", "FewShot_exp_N2.pt"),
             ("AE fine-tuned on 163 original exp TMs (section 6 recipe)", "FT_best_weights_N163.pt")]
PCA_MODELS = [("PCA fitted on 163 sim TMs", "sim", 163),
              ("PCA fitted on 2 original exp TMs", "exp", 2), ("PCA fitted on 163 original exp TMs", "exp", 163)]
print("Frozen model list (seed %d):" % args.seed)
for name, f in AE_MODELS:
    print(f"  AE   {name:60s} {f}{'' if (AE_DIR / f).exists() else '   <-- MISSING'}")
for name, kind, n in PCA_MODELS:
    print(f"  PCA  {name}")
if args.plan:
    sys.exit(0)
if LOCK.exists() and not args.allow_rerun:
    sys.exit(f"{LOCK.name} exists: the test set has already been scored. Use --allow-rerun only if you will report it as a rerun.")

device = torch.device(args.device)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)

exp_stack = paths.load_exp_stack()
exp_train = np.load(EXP_DIR / "train_split_idx.npy")
TEST = {"original exp": torch.tensor(exp_stack[np.load(EXP_DIR / "test_split_idx.npy")], dtype=torch.cfloat)}
print({k: len(v) for k, v in TEST.items()}, "test TMs")


def load_exp(idx):
    return torch.tensor(np.ascontiguousarray(exp_stack[idx]), dtype=torch.cfloat)


def to_patterns(tm):
    p = tm.permute(0, 2, 1).reshape(-1, PIXELS)
    return torch.cat([p.real, p.imag], dim=1).float()


def to_tm(patterns, n_tm):
    c = torch.complex(patterns[:, :PIXELS], patterns[:, PIXELS:])
    return c.reshape(n_tm, N_IN, PIXELS).permute(0, 2, 1)


def per_tm_error(recon, true):
    return phase_aligned_loss_wholeTM_perSample(recon, true)[1].item()


@torch.no_grad()
def ae_error(model, tms):
    errs = []
    for i in range(len(tms)):                      # one TM at a time: exact mean over TMs
        X = tms[i:i + 1].to(device)
        X_in = torch.stack([X.real, X.imag], dim=1).reshape(1, 2, 205, 205, N_IN).permute(0, 1, 4, 2, 3)
        pred, _ = model(F.pad(X_in, (0, 3, 0, 3)), show_shapes=False)
        pred = pred[:, :, :, :205, :205].permute(0, 1, 3, 4, 2).reshape(1, 2, PIXELS, N_IN)
        errs.append(per_tm_error(torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()), X))
    return float(np.mean(errs))


def fit_pca(X, k=K62):
    X = X.to(device)
    mean = X.mean(0, keepdim=True)
    X -= mean
    w, U = torch.linalg.eigh((X @ X.T).double())
    w, U = w.flip(0)[:k], U.flip(1)[:, :k]
    U = U[:, w > 1e-6 * w[0]]                      # few TMs: drop numerical-noise directions
    V = X.T @ U.float()
    V /= V.norm(dim=0, keepdim=True)
    return mean, V


@torch.no_grad()
def pca_error(mean, V, tms):
    errs = []
    for i in range(len(tms)):
        tm = tms[i:i + 1]
        rec = to_tm((to_patterns(tm).to(device) - mean) @ V @ V.T + mean, 1).cpu()
        errs.append(per_tm_error(rec, tm))
    return float(np.mean(errs))


def sim_patterns(n_tm, seed=0):
    """Composite-5m simulated TMs, same forward model and noise as pretraining (copy of the function in pca_baseline.py)."""
    sim_dir = paths.SIM_DIR
    sim_idx = np.load(paths.SIM_SPLIT_DIR / "train_composite5m_idx_2000.npy")[:n_tm]
    unique_ids = np.unique(sim_idx.ravel()).astype(np.int64)
    pos_of = {int(i): n for n, i in enumerate(unique_ids)}
    rows = np.array([[pos_of[int(i)] for i in row] for row in sim_idx])
    pool = []
    for i in unique_ids:
        stem = f"TM{int(i):04d}"
        path = next((p for p in (sim_dir / f"{stem}.pkl", sim_dir / f"{stem}-GPU.pkl") if p.is_file()), None)
        with open(path, "rb") as f:
            pool.append(torch.from_numpy(pickle.load(f)["tm_mode"]).to(torch.complex64))
    pool = torch.stack(pool).to(device)
    with open(sim_dir / "dist_pim_to_real.pkl", "rb") as f:
        dist = torch.tensor(pickle.load(f)).to(device=device, dtype=torch.complex64)
    with open(sim_dir / "prox_real_to_pim.pkl", "rb") as f:
        prox = torch.from_numpy(pickle.load(f)).to(device=device, dtype=torch.complex64)
    u_in = torch.tensor(np.load(paths.INPUTS_FILE), dtype=torch.complex64, device=device)
    a_in = prox @ u_in
    a_in = a_in / (torch.linalg.vector_norm(a_in, dim=0, keepdim=True) + 1e-8)
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for s in range(0, n_tm, 4):
        r = rows[s:s + 4]
        T = pool[r[:, 0]]
        for j in range(1, 5):
            T = pool[r[:, j]] @ T
        X = dist @ (T @ a_in)
        B = X.shape[0]
        an = torch.empty(B, 1, 1, device=device).uniform_(0.01, 0.15, generator=g)
        pn = torch.empty(B, 1, 1, device=device).uniform_(0.01, 0.1, generator=g)
        amp, ph = X.abs(), X.angle()
        amp = amp + an * amp * torch.randn(amp.shape, device=device, generator=g)
        ph = ph + pn * torch.randn(ph.shape, device=device, generator=g)
        X = amp * torch.exp(1j * ph)
        X = X / (torch.linalg.svdvals(X)[:, 0][:, None, None] + 1e-8)
        out.append(to_patterns(X.cpu()))
    return torch.cat(out)


import json
hp = json.loads((AE_DIR / "hyperparameters.json").read_text())["params"]
results = []                                        # (model, dataset, error, n_tm)


def record(name, errs):
    for ds, e in errs.items():
        results.append((name, ds, e, len(TEST[ds])))
        print(f"  {name:60s} {ds:13s} {100 * e:6.2f}%")


for name, f in AE_MODELS:
    if not (AE_DIR / f).exists():
        print(f"  skipped (missing weights): {name}")
        continue
    m = TM_Autoencoder(enc_channels=[hp["enc1"], 2 * hp["enc1"], 3 * hp["enc1"]], latent_dim=8,
                       attn_heads=hp["attn_heads"], linformer_k=hp["linformer_k"]).to(device)
    m.load_state_dict(torch.load(AE_DIR / f, map_location=device))
    m.eval()
    record(name, {ds: ae_error(m, tms) for ds, tms in TEST.items()})
    del m
    torch.cuda.empty_cache()

for name, kind, n in PCA_MODELS:
    if kind == "sim":
        X_fit = sim_patterns(n)
    else:
        X_fit = to_patterns(load_exp(exp_train[:n]))
    mean, V = fit_pca(X_fit)
    del X_fit
    record(name, {ds: pca_error(mean, V, tms) for ds, tms in TEST.items()})
    del mean, V
    torch.cuda.empty_cache()

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "dataset", "test_error", "n_tm"])
    w.writerows([(m, d, round(e, 5), n) for m, d, e, n in results])
LOCK.write_text("final test evaluation done" + (" (RERUN)" if args.allow_rerun else "") + f", seed {args.seed}\n")
print("saved", OUT)
