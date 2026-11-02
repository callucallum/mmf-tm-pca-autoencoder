import argparse
import copy
import csv
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS; must be set before CUDA starts

import numpy as np
import optuna
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths
from autoencoder_functions import (
    TM_Autoencoder,
    phase_aligned_loss_wholeTM_perSample,
)

# Usage: python Final_62x_ES_FT.py --trials 0 16 7 --seeds 42 43 44 [--exp-ft-n 10 163] [--eval-test]
# With several --exp-ft-n values each (trial, seed) is pretrained once and fine-tuned separately for every size.
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, nargs="+", required=True, help="Optuna trial numbers to take hyperparameters from (0 = baseline recipe)")
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
parser.add_argument("--exp-ft-n", "--exp-train-n", type=int, nargs="+", default=[10],
                    help="exp TMs for fine-tuning (5/10/25/50/100/125/150, or 163 for all); with --exp-only: the number of exp TMs each model is trained on from scratch")
parser.add_argument("--n-sim", type=int, default=2000)
parser.add_argument("--study", default="TMM_62x_ES_FT_v2")
parser.add_argument("--max-pretrain-epochs", type=int, default=1000, help="epoch cap (the Optuna study capped at 150)")
parser.add_argument("--max-ft-epochs", type=int, default=300)
parser.add_argument("--pretrain-patience", type=int, default=30, help="epochs without improvement before pretraining stops (30 for the main runs)")
parser.add_argument("--out-name", default="final_62x", help="output folder name under TM_OUT_ROOT (the released weights use final_62x, and exp_only_62x for the --exp-only runs)")
parser.add_argument("--exp-only", action="store_true",
                    help="matched exp-only baseline: same architecture and recipe as the pretraining stage (pretrain_lr, weight_decay, pretrain batch size, "
                         "patience, --max-pretrain-epochs) but trained from scratch on --exp-ft-n exp TMs, with no simulation; checkpoint chosen on the 20 exp val TMs")
parser.add_argument("--exp-batch-size", type=int, default=0,
                    help="--exp-only: batch size for training from scratch (0 = the trial's pretrain batch size). The baseline-recipe exp-only runs and the fine-tuning stage use 2")
parser.add_argument("--dead-plateau-epoch", type=int, default=100, help="stop a run still at ~100%% exp-val error after this many epochs (exp-only runs can escape later: try 200)")
parser.add_argument("--eval-test", action="store_true", help="also score the final model on the 21 exp test TMs (off by default)")
parser.add_argument("--smoke", action="store_true", help="one tiny run to check the pipeline works; writes nothing")
parser.add_argument("--fast", action="store_true", help="turn off the deterministic settings (faster, but reruns will not match exactly)")
args = parser.parse_args()

LATENT_DIM = 8                 # 62x compression
PRETRAIN_PATIENCE = args.pretrain_patience
FT_PATIENCE = 30
DEAD_PLATEAU_EPOCH = args.dead_plateau_epoch   # stop a run still at ~100% exp-val error after this many epochs
EVAL_BATCH = 4
if args.smoke:
    args.n_sim, args.max_pretrain_epochs, args.max_ft_epochs = 24, 2, 2
    args.trials, args.seeds = args.trials[:1], args.seeds[:1]

SIM_DIR = paths.SIM_DIR
EXP_SPLIT_DIR = paths.EXP_SPLIT_DIR
EXP_DATASET = paths.EXP_DATA
DB = paths.OPTUNA_DB
OUT_ROOT = paths.OUT_ROOT / args.out_name
RESULTS_CSV = OUT_ROOT / "final_results.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.fast:
    torch.backends.cudnn.benchmark = True
else:
    # Same seed -> same result on this machine: fixed conv algorithms, deterministic kernels, no TF32 rounding.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# ── Hyperparameters from the Optuna study ────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name=args.study, storage=f"sqlite:///{DB.as_posix()}")
trial_by_number = {t.number: t for t in study.trials}
missing = [n for n in args.trials if n not in trial_by_number or not trial_by_number[n].params]
if missing:
    raise ValueError(f"Trials not found (or without parameters) in study '{args.study}': {missing}")

# ── Data: everything loaded once (the test split only if --eval-test) ────────
if not args.exp_only:                                            # the exp-only baseline uses no simulation at all
    sim_idx = np.load(paths.SIM_SPLIT_DIR / f"train_composite5m_idx_{2000}.npy")[:args.n_sim]
    unique_ids = np.unique(sim_idx.ravel()).astype(np.int64)
    pos_of = {int(i): n for n, i in enumerate(unique_ids)}
    sim_rows = np.array([[pos_of[int(i)] for i in row] for row in sim_idx])

    pool = []
    for i in unique_ids:
        stem = f"TM{int(i):04d}"
        path = next((p for p in (SIM_DIR / f"{stem}.pkl", SIM_DIR / f"{stem}-GPU.pkl") if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No pickle found for {stem}")
        with open(path, "rb") as f:
            pool.append(torch.from_numpy(pickle.load(f)["tm_mode"]).to(torch.complex64))
    pool = torch.stack(pool)  # [P, 386, 386] on CPU; composites are built per batch on the GPU
    print(f"Loaded {len(pool)} 1m mode-basis TMs for {len(sim_rows)} composite training TMs")

    with open(SIM_DIR / "dist_pim_to_real.pkl", "rb") as f:
        dist_pim_to_real = torch.tensor(pickle.load(f)).to(device=device, dtype=torch.complex64)
    with open(SIM_DIR / "prox_real_to_pim.pkl", "rb") as f:
        prox_real_to_pim = torch.from_numpy(pickle.load(f)).to(device=device, dtype=torch.complex64)
    u_in = torch.tensor(np.load(paths.INPUTS_FILE),
                        dtype=torch.complex64, device=device)
    a_in = prox_real_to_pim @ u_in
    a_in_i = a_in / (torch.linalg.vector_norm(a_in, dim=0, keepdim=True) + 1e-8)

exp_stack = np.load(EXP_DATASET, allow_pickle=True, mmap_mode="r")["TM"]
TM_val = torch.tensor(exp_stack[np.load(EXP_SPLIT_DIR / "val_split_idx.npy")], dtype=torch.cfloat, device=device)
TM_test = None
if args.eval_test:
    TM_test = torch.tensor(exp_stack[np.load(EXP_SPLIT_DIR / "test_split_idx.npy")], dtype=torch.cfloat, device=device)
print(f"Exp val: {len(TM_val)}, fine-tuning sizes: {args.exp_ft_n}" + (f", exp test: {len(TM_test)}" if TM_test is not None else ""))


def load_ft_set(n):
    """First n exp training TMs (nested subsets); kept on the CPU and moved to the GPU per batch."""
    ft_file = "train_split_idx.npy" if n == 163 else f"train_split_idx_{n}.npy"
    if (EXP_SPLIT_DIR / ft_file).exists():
        idx = np.load(EXP_SPLIT_DIR / ft_file)
    else:                                            # sizes without a saved split (1, 2, 3): the splits are nested prefixes of the train list
        idx = np.load(EXP_SPLIT_DIR / "train_split_idx.npy")[:n]
    return torch.tensor(exp_stack[idx], dtype=torch.cfloat)


def sim_batch(batch_rows):
    """Composite 5m TMs through the same forward model + noise as the PT/ES scripts."""
    rows = sim_rows[batch_rows]
    T = pool[rows[:, 0]].to(device)
    for j in range(1, 5):
        T = pool[rows[:, j]].to(device) @ T
    X = dist_pim_to_real @ (T @ a_in_i)
    B = X.shape[0]
    amp_noise = torch.empty(B, 1, 1, device=device).uniform_(0.01, 0.15)
    phase_noise = torch.empty(B, 1, 1, device=device).uniform_(0.01, 0.1)
    amp, phase = X.abs(), X.angle()
    amp = amp + amp_noise * amp * torch.randn_like(amp)
    phase = phase + phase_noise * torch.randn_like(phase)
    X = amp * torch.exp(1j * phase)
    return X / (torch.linalg.svdvals(X)[:, 0][:, None, None] + 1e-8)


def forward_loss(model, X):
    B = X.shape[0]
    X_in = torch.stack([X.real, X.imag], dim=1).reshape(B, 2, 205, 205, 145).permute(0, 1, 4, 2, 3)
    pred, _ = model(F.pad(X_in, (0, 3, 0, 3)), show_shapes=False)
    pred = pred[:, :, :, :205, :205].permute(0, 1, 3, 4, 2).reshape(B, 2, 42025, 145)
    pred_complex = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())
    return phase_aligned_loss_wholeTM_perSample(pred_complex, X)[0]


def train_step(model, optimizer, X):
    optimizer.zero_grad(set_to_none=True)
    loss = forward_loss(model, X)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, TMs):
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(TMs), EVAL_BATCH):
        X = TMs[s:s + EVAL_BATCH]
        total += forward_loss(model, X).item() * len(X)
        n += len(X)
    return total / n


def run_one(trial_no, seed, ns):
    params = trial_by_number[trial_no].params
    enc1, heads, k = int(params["enc1"]), int(params["attn_heads"]), int(params["linformer_k"])
    pre_lr, ft_lr, wd = float(params["pretrain_lr"]), float(params["finetune_lr"]), float(params["weight_decay"])
    pre_bs = int(params["pretrain_batch_size"])

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    tag = f"[trial {trial_no} | seed {seed}]"
    model = TM_Autoencoder(enc_channels=[enc1, 2 * enc1, 3 * enc1], latent_dim=LATENT_DIM,
                           attn_heads=heads, linformer_k=k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=pre_lr, weight_decay=wd)

    # Stage 1: sim training, checkpoint chosen on the 20 exp-val TMs (early stopping on experimental validation TMs)
    best_val, best_epoch, best_state, bad = float("inf"), -1, None, 0
    t0 = time.time()
    for epoch in range(args.max_pretrain_epochs):
        model.train()
        perm = rng.permutation(len(sim_rows))
        for s in range(0, len(perm), pre_bs):
            train_step(model, optimizer, sim_batch(perm[s:s + pre_bs]))
        val = evaluate(model, TM_val)
        print(f"{tag} pretrain epoch {epoch}: exp-val {val:.4f}, best {min(best_val, val):.4f}, {time.time() - t0:.0f}s elapsed")
        if val < best_val:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PRETRAIN_PATIENCE:
                break
        if epoch >= DEAD_PLATEAU_EPOCH and best_val > 0.98:
            print(f"{tag} still at ~100% after {epoch} epochs: stopping as a dead pretrain")
            break
    es_val = best_val

    # Snapshot the end of pretraining so every fine-tuning size starts from exactly the same state
    opt_state = copy.deepcopy(optimizer.state_dict())
    rng_state = rng.bit_generator.state
    out = OUT_ROOT / f"trial{trial_no}" / (f"seed{seed}" if PRETRAIN_PATIENCE == 30 else f"seed{seed}_pat{PRETRAIN_PATIENCE}")
    if not args.smoke:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out / "ES_best_weights.pt")
        (out / "hyperparameters.json").write_text(json.dumps({"trial": trial_no, "seed": seed, "params": params, **vars(args)}, indent=2, default=str))

    rows = []
    for n in ns:
        # Stage 2: fine-tune on n exp TMs (best weights + last-epoch optimiser state, as in the ES_FT scripts)
        TM_ft = load_ft_set(n)
        model.load_state_dict(best_state)
        optimizer.load_state_dict(copy.deepcopy(opt_state))
        for g in optimizer.param_groups:
            g["lr"] = ft_lr
        rng.bit_generator.state = rng_state
        ft_best, ft_best_epoch, ft_best_state, bad = float("inf"), -1, None, 0
        for epoch in range(args.max_ft_epochs):
            model.train()
            perm = rng.permutation(len(TM_ft))
            for s in range(0, len(perm), 2):
                train_step(model, optimizer, TM_ft[perm[s:s + 2]].to(device))
            val = evaluate(model, TM_val)
            if val < ft_best:
                ft_best, ft_best_epoch, bad = val, epoch, 0
                ft_best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= FT_PATIENCE:
                    break
        test_error = ""
        if TM_test is not None:
            model.load_state_dict(ft_best_state)
            test_error = evaluate(model, TM_test)
        print(f"{tag} [{n} exp TMs] ES val {es_val:.4f} -> FT val {ft_best:.4f}" + (f", test {test_error:.4f}" if test_error != "" else ""))
        if not args.smoke:
            torch.save(ft_best_state, out / f"FT_best_weights_N{n}.pt")
        rows.append({"trial": trial_no, "seed": seed, "exp_ft_n": n, "n_sim": args.n_sim,
                     "es_val": round(es_val, 5), "ft_val": round(ft_best, 5),
                     "test_error": test_error if test_error == "" else round(test_error, 5),
                     "es_best_epoch": best_epoch, "ft_best_epoch": ft_best_epoch,
                     "optuna_value": trial_by_number[trial_no].value, "params": json.dumps(params),
                     "pretrain_patience": PRETRAIN_PATIENCE})
        del TM_ft, ft_best_state

    del model, optimizer, best_state, opt_state
    torch.cuda.empty_cache()
    return rows


def run_exp_only(trial_no, seed, ns):
    """Matched exp-only baseline: trial architecture, pretrain_lr / weight_decay / pretrain batch size, patience and epoch cap,
    trained from scratch on the first n exp training TMs (no simulation). Same init for every n (same seed)."""
    params = trial_by_number[trial_no].params
    enc1, heads, k = int(params["enc1"]), int(params["attn_heads"]), int(params["linformer_k"])
    lr, wd = float(params["pretrain_lr"]), float(params["weight_decay"])
    bs = args.exp_batch_size or int(params["pretrain_batch_size"])
    tag = f"[exp-only | trial {trial_no} | seed {seed} | batch {bs}]"
    rows = []
    for n in ns:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)
        model = TM_Autoencoder(enc_channels=[enc1, 2 * enc1, 3 * enc1], latent_dim=LATENT_DIM,
                               attn_heads=heads, linformer_k=k).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        TM_tr = load_ft_set(n)
        best_val, best_epoch, best_state, bad = float("inf"), -1, None, 0
        t0 = time.time()
        for epoch in range(args.max_pretrain_epochs):
            model.train()
            perm = rng.permutation(len(TM_tr))
            for s in range(0, len(perm), bs):
                train_step(model, optimizer, TM_tr[perm[s:s + bs]].to(device))
            val = evaluate(model, TM_val)
            print(f"{tag} [{n} exp TMs] epoch {epoch}: exp-val {val:.4f}, best {min(best_val, val):.4f}, {time.time() - t0:.0f}s elapsed")
            if val < best_val:
                best_val, best_epoch, bad = val, epoch, 0
                best_state = {n_: v.detach().cpu().clone() for n_, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PRETRAIN_PATIENCE:
                    break
            if epoch >= DEAD_PLATEAU_EPOCH and best_val > 0.98:
                print(f"{tag} [{n} exp TMs] still at ~100% after {epoch} epochs: stopping as a dead run")
                break
        test_error = ""
        if TM_test is not None:
            model.load_state_dict(best_state)
            test_error = evaluate(model, TM_test)
        print(f"{tag} [{n} exp TMs] best exp-val {best_val:.4f} at epoch {best_epoch}" + (f", test {test_error:.4f}" if test_error != "" else ""))
        out = OUT_ROOT / f"trial{trial_no}" / f"seed{seed}"
        if not args.smoke:
            out.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, out / f"ExpOnly_best_weights_N{n}.pt")
            (out / "hyperparameters.json").write_text(json.dumps({"trial": trial_no, "seed": seed, "params": params, "mode": "exp-only", **vars(args)}, indent=2, default=str))
        rows.append({"trial": trial_no, "seed": seed, "exp_ft_n": n, "n_sim": 0,
                     "es_val": "", "ft_val": round(best_val, 5),
                     "test_error": test_error if test_error == "" else round(test_error, 5),
                     "es_best_epoch": "", "ft_best_epoch": best_epoch,
                     "optuna_value": trial_by_number[trial_no].value, "params": json.dumps(params),
                     "pretrain_patience": PRETRAIN_PATIENCE})
        del model, optimizer, best_state, TM_tr
        torch.cuda.empty_cache()
    return rows


# ── Run every (trial, seed) pair, skipping any (trial, seed, size) already in the results file ───
done = set()
if not args.smoke and RESULTS_CSV.exists():
    with open(RESULTS_CSV, newline="") as f:
        done = {(int(r["trial"]), int(r["seed"]), int(r["exp_ft_n"]), int(r["n_sim"]), int(r.get("pretrain_patience") or 30))
                for r in csv.DictReader(f)}


def append_result(row):
    """Append to the results CSV; first upgrades an older file that has no pretrain_patience column (those rows used 30)."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            fields, old_rows = reader.fieldnames, list(reader)
        if "pretrain_patience" not in fields:
            for r in old_rows:
                r["pretrain_patience"] = 30
            with open(RESULTS_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows([{k: r.get(k, "") for k in row} for r in old_rows])
    new_file = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


for trial_no in args.trials:
    for seed in args.seeds:
        n_sim_key = 0 if args.exp_only else args.n_sim
        ns = [n for n in args.exp_ft_n if (trial_no, seed, n, n_sim_key, PRETRAIN_PATIENCE) not in done]
        if not ns:
            print(f"[trial {trial_no} | seed {seed}] all requested sizes already in {RESULTS_CSV.name}, skipping")
            continue
        for row in (run_exp_only if args.exp_only else run_one)(trial_no, seed, ns):
            if not args.smoke:
                append_result(row)
