import argparse
import pickle
import random
import sys
import time
import warnings
from pathlib import Path

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

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="one tiny in-memory trial to check the pipeline runs")
args = parser.parse_args()

# ── Settings ─────────────────────────────────────────────────────────────────
SEED = 42
LATENT_DIM = 8                 # 62x compression - fixed, not tuned
N_SIM = 2000                   # composite 5m sim TMs for pretraining
EXP_FT_N = 10                  # exp TMs used for fine-tuning
PRETRAIN_MAX_EPOCHS = 150
PRETRAIN_PATIENCE = 30
FT_MAX_EPOCHS = 300
FT_PATIENCE = 30
PRUNE_WARMUP = 40              # epochs before the median pruner may act (the model can sit on a ~100% plateau first)
DEAD_PLATEAU_EPOCH = 80        # prune a trial still at ~100% exp-val error after this many epochs
N_TRIALS = 25
EVAL_BATCH = 4

if args.smoke:
    N_SIM, PRETRAIN_MAX_EPOCHS, FT_MAX_EPOCHS, N_TRIALS = 24, 2, 2, 1

# The channel attention embeds the 13x13 map, so embed_dim = 169 and num_heads must divide it: only 1 or 13 work.
ENC1_CHOICES = [8, 16, 24, 32]
HEADS_CHOICES = [1, 13]
K_CHOICES = [4, 8, 16, 24, 32]
PRE_BS_CHOICES = [4, 8, 16]
PRE_LR_RANGE = (3e-4, 3e-3)
FT_LR_RANGE = (1e-5, 3e-3)
WD_RANGE = (5e-7, 5e-5)
STUDY_NAME = "TMM_62x_ES_FT_v2"
OLD_STUDY_NAME = "TMM_62x_ES_FT"      # first study (heads in {1,2,4}); its valid trials are copied across

BASELINE_PARAMS = {
    "enc1": 16, "attn_heads": 1, "linformer_k": 16,
    "pretrain_lr": 0.001529,
    "finetune_lr": 0.001529,   # the fine-tuning scripts load the pretrain optimiser state, which restores lr=0.001529
    "weight_decay": 1.108781e-06,
    "pretrain_batch_size": 8,
}

SIM_DIR = paths.SIM_DIR
EXP_SPLIT_DIR = paths.EXP_SPLIT_DIR
EXP_DATASET = paths.EXP_DATA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ── Data (the test split is never loaded here) ───────────────────────────────
sim_idx = np.load(paths.SIM_SPLIT_DIR / f"train_composite5m_idx_{2000}.npy")[:N_SIM]
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
exp_val_idx = np.load(EXP_SPLIT_DIR / "val_split_idx.npy")
exp_ft_idx = np.load(EXP_SPLIT_DIR / f"train_split_idx_{EXP_FT_N}.npy")
TM_val = torch.tensor(exp_stack[exp_val_idx], dtype=torch.cfloat, device=device)
TM_ft = torch.tensor(exp_stack[exp_ft_idx], dtype=torch.cfloat, device=device)
print(f"Exp val: {len(TM_val)}, exp fine-tune: {len(TM_ft)}")


def sim_batch(batch_rows):
    """Composite 5m TMs for a batch, pushed through the same forward model + noise as the PT/ES scripts."""
    rows = sim_rows[batch_rows]
    T = pool[rows[:, 0]].to(device)
    for j in range(1, 5):
        T = pool[rows[:, j]].to(device) @ T
    X = dist_pim_to_real @ (T @ a_in_i)                       # [B, 42025, 145]
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
    pred, _ = model(F.pad(X_in, (0, 3, 0, 3)))
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


def objective(trial):
    enc1 = trial.suggest_categorical("enc1", ENC1_CHOICES)
    heads = trial.suggest_categorical("attn_heads", HEADS_CHOICES)
    k = trial.suggest_categorical("linformer_k", K_CHOICES)
    pre_lr = trial.suggest_float("pretrain_lr", *PRE_LR_RANGE, log=True)
    ft_lr = trial.suggest_float("finetune_lr", *FT_LR_RANGE, log=True)
    wd = trial.suggest_float("weight_decay", *WD_RANGE, log=True)
    pre_bs = trial.suggest_categorical("pretrain_batch_size", PRE_BS_CHOICES)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    model = TM_Autoencoder(enc_channels=[enc1, 2 * enc1, 3 * enc1], latent_dim=LATENT_DIM,
                           attn_heads=heads, linformer_k=k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=pre_lr, weight_decay=wd)

    # Stage 1: sim training, checkpoint chosen on the 20 exp-val TMs (early stopping on experimental validation TMs)
    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(PRETRAIN_MAX_EPOCHS):
        model.train()
        perm = rng.permutation(len(sim_rows))
        for s in range(0, len(perm), pre_bs):
            train_step(model, optimizer, sim_batch(perm[s:s + pre_bs]))
        val = evaluate(model, TM_val)
        print(f"[Trial {trial.number}] pretrain epoch {epoch}: exp-val {val:.4f}, best {min(best_val, val):.4f}, {time.time() - t0:.0f}s elapsed")
        if val < best_val:
            best_val, bad = val, 0
            best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PRETRAIN_PATIENCE:
                break
        trial.report(best_val, epoch)
        if trial.should_prune() or (epoch >= DEAD_PLATEAU_EPOCH and best_val > 0.98):
            raise optuna.TrialPruned()
    trial.set_user_attr("es_val_error", best_val)

    # Stage 2: fine-tune on EXP_FT_N exp TMs, as the ES_FT scripts do (best weights + last-epoch optimiser state)
    model.load_state_dict(best_state)
    for g in optimizer.param_groups:
        g["lr"] = ft_lr
    ft_best, bad = float("inf"), 0
    for epoch in range(FT_MAX_EPOCHS):
        model.train()
        perm = rng.permutation(len(TM_ft))
        for s in range(0, len(perm), 2):
            train_step(model, optimizer, TM_ft[perm[s:s + 2]])
        val = evaluate(model, TM_val)
        if val < ft_best:
            ft_best, bad = val, 0
        else:
            bad += 1
            if bad >= FT_PATIENCE:
                break
    trial.set_user_attr("ft_val_error", ft_best)
    print(f"[Trial {trial.number}] ES val {best_val:.4f} -> FT val {ft_best:.4f}")

    del model, optimizer, best_state
    torch.cuda.empty_cache()
    return ft_best


warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
if args.smoke:
    study = optuna.create_study(direction="minimize")
else:
    db = paths.OPTUNA_DB
    db.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db.as_posix()}"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=PRUNE_WARMUP, interval_steps=5),
    )
    if len(study.trials) == 0:
        try:
            old = optuna.load_study(study_name=OLD_STUDY_NAME, storage=storage)
            dists = {
                "enc1": optuna.distributions.CategoricalDistribution(ENC1_CHOICES),
                "attn_heads": optuna.distributions.CategoricalDistribution(HEADS_CHOICES),
                "linformer_k": optuna.distributions.CategoricalDistribution(K_CHOICES),
                "pretrain_lr": optuna.distributions.FloatDistribution(*PRE_LR_RANGE, log=True),
                "finetune_lr": optuna.distributions.FloatDistribution(*FT_LR_RANGE, log=True),
                "weight_decay": optuna.distributions.FloatDistribution(*WD_RANGE, log=True),
                "pretrain_batch_size": optuna.distributions.CategoricalDistribution(PRE_BS_CHOICES),
            }
            for t in old.trials:
                if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) and t.params.get("attn_heads") in HEADS_CHOICES:
                    study.add_trial(optuna.trial.create_trial(
                        state=t.state, params=t.params, distributions=dists,
                        value=t.value if t.state == optuna.trial.TrialState.COMPLETE else None,
                        user_attrs=t.user_attrs, intermediate_values=t.intermediate_values))
            print(f"Copied {len(study.trials)} valid trials from '{OLD_STUDY_NAME}'")
        except KeyError:
            pass
if len(study.trials) == 0:
    study.enqueue_trial(BASELINE_PARAMS)   # evaluate the current recipe first as the reference
study.optimize(objective, n_trials=N_TRIALS)

best = study.best_trial
print(f"Best trial {best.number}: FT val error {best.value:.4f}  params {best.params}")
