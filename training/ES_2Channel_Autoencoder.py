from datetime import datetime
import os
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import psutil
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths
from autoencoder_functions import (
    TM_Autoencoder,
    phase_aligned_loss_wholeTM_perSample,
)

seed_value = 44
tm_dataset_size = 2000

SIM_DATA_DIR = paths.SIM_DIR
EXP_DATASPLIT_DIR = paths.EXP_SPLIT_DIR
MODEL_DIR = paths.OUT_ROOT / "5m_fibre" / "62x_compression" / f"seed{seed_value}" / f"{tm_dataset_size}"

rng = np.random.default_rng(seed_value)
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # for GPU
torch.backends.cudnn.deterministic = True  # ensure deterministic operations
torch.backends.cudnn.benchmark = False

# the following code is related to dataloader reproducibility (see: https://pytorch.org/docs/stable/notes/randomness.html)
# as of 17/03 num_workers = 0 in the main code, therefore the seed_worker function is never used
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(seed_value)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def svd_norm_torch(tm):
    S = torch.linalg.svdvals(tm)
    return tm / (S[0] + 1e-8)

epoch = int(sys.argv[1])
sub_batch = int(sys.argv[2])

# ── Sim training data (composite 5m TMs, same as PT) ──────────────────────────
train_split_idx = np.load(paths.SIM_SPLIT_DIR / f"train_composite5m_idx_{tm_dataset_size}.npy")

if train_split_idx.ndim != 2 or train_split_idx.shape[1] != 5:
    raise ValueError(f"Expected training indices with shape [N, 5], got {train_split_idx.shape}")

batch_size_data = 100
num_sub_batches = math.ceil(len(train_split_idx) / batch_size_data)

start_idx = sub_batch * batch_size_data
end_idx = min((sub_batch + 1) * batch_size_data, len(train_split_idx))

train_idx_sub = train_split_idx[start_idx:end_idx]
required_tm_indices = np.unique(train_idx_sub.ravel()).astype(np.int64)
required_tm_index_set = set(required_tm_indices.tolist())

print(f"Sub-batch {sub_batch}: train_idx[{start_idx}:{end_idx}] -> {len(train_idx_sub)} samples")

# Load only the sim TM files needed for this training sub-batch.
tmm_tm_mode = {}
for dataset_idx in required_tm_index_set:
    filename_stem = f"TM{int(dataset_idx):04d}"
    candidate_paths = (
        SIM_DATA_DIR / f"{filename_stem}.pkl",
        SIM_DATA_DIR / f"{filename_stem}-GPU.pkl",
    )
    file_path = next((path for path in candidate_paths if path.is_file()), None)

    try:
        if file_path is None:
            expected_files = ", ".join(path.name for path in candidate_paths)
            raise FileNotFoundError(f"Could not find any of: {expected_files}")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            tmm_tm_mode[int(dataset_idx)] = torch.from_numpy(data["tm_mode"]).to(device)
    except Exception as e:
        print(f"Error processing dataset index {dataset_idx}: {e}")
        raise

print(f"Loaded {len(tmm_tm_mode)}/{len(required_tm_index_set)} required sim TM files to GPU")

def build_composite_tm(row_indices, tm_pool):
    """row_indices: array of shape [5], tm_pool: full tmm_tm_mode tensor/list"""
    assert len(row_indices) == 5, f"Expected 5 TM indices, got {len(row_indices)}"
    T = tm_pool[row_indices[0]]
    for idx in row_indices[1:]:
        T = tm_pool[idx] @ T  # fix propagation order convention here
    return T

TM_train_sub = torch.stack([
    build_composite_tm(row, tmm_tm_mode) for row in train_idx_sub
])

with open(SIM_DATA_DIR / 'dist_pim_to_real.pkl', 'rb') as f:
    dist_pim_to_real = pickle.load(f)
    dist_pim_to_real = torch.tensor(dist_pim_to_real).to(device)

with open(SIM_DATA_DIR / 'prox_real_to_pim.pkl', 'rb') as f:
    prox_real_to_pim = pickle.load(f)
    prox_real_to_pim = torch.from_numpy(prox_real_to_pim).to(device)

n_points_modes_371 = 371
num_inputs = 145

u_in_2d = np.load(paths.INPUTS_FILE)
u_in = torch.tensor(u_in_2d, dtype=torch.complex64, device=device)

del tmm_tm_mode
torch.cuda.empty_cache()

a_in = prox_real_to_pim @ u_in        # (386,145)
a_in_i = a_in / (torch.linalg.vector_norm(a_in, dim=0, keepdim=True) + 1e-8)

def calculate_forward_tm(tmm_tm_i):
    # Propagate
    a_out = tmm_tm_i @ a_in_i

    # Modes -> pixels
    TM_pixel = dist_pim_to_real @ a_out

    amp_noise = torch.empty(1).uniform_(0.01, 0.15).item()
    phase_noise = torch.empty(1).uniform_(0.01, 0.1).item()

    # --- Add Gaussian amplitude & phase noise ---
    amp = torch.abs(TM_pixel)
    phase = torch.angle(TM_pixel)

    amp += amp_noise * amp * torch.randn_like(amp)
    phase += phase_noise * torch.randn_like(phase)

    noisy_TM = amp * torch.exp(1j * phase)

    return svd_norm_torch(noisy_TM)

TM_train_list = []
for tm_x in TM_train_sub:
    TM_train_list.append(calculate_forward_tm(tm_x))
TM_train_list = torch.stack(TM_train_list)

del dist_pim_to_real, prox_real_to_pim

print(f"Found {len(TM_train_list)} sim training files, TM shape: {TM_train_list[0].shape}")

# ── Real exp validation data (20 TMs, used ONLY for early stopping - no gradient updates) ──
exp_val_idx = np.load(EXP_DATASPLIT_DIR / "val_split_idx.npy")
exp_dataset = np.load(paths.EXP_DATA, allow_pickle=True, mmap_mode="r")
TM_exp_val = torch.tensor(exp_dataset['TM'][exp_val_idx], dtype=torch.cfloat, device=device)

print(f"Found {len(TM_exp_val)} real exp validation TMs (used for early stopping only)")

print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

batch_size = 8  # sim training batch size, matches PT
exp_val_batch_size = 2  # matches FT's real-data val batch size
best_val_loss = float("inf")
epochs_no_improve = 0
pretrain_lr = 0.001529  # same LR regime as PT - this is sim-only training, not fine-tuning pretrained weights

if epoch == 0 and sub_batch == 0:
    torch.manual_seed(seed_value)  # re-seed right before init: PT consumes extra RNG draws building
    torch.cuda.manual_seed_all(seed_value)  # a sim val set that ES doesn't, which would otherwise shift
    model = TM_Autoencoder(latent_dim=8).to(device)  # the weight-init RNG stream vs PT and break comparability
    optimizer = optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=1.108781e-06)
    print("Model initialised from scratch")
else:
    model = TM_Autoencoder(latent_dim=8).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=1.108781e-06)
    optimizer.load_state_dict(torch.load(MODEL_DIR / "ES_2Channel_optimiser.pt"))
    model.load_state_dict(torch.load(MODEL_DIR / "ES_2Channel_int_weights.pt"))
    print("Imported most recent model weights and optimiser")

if epoch > 0 or sub_batch > 0:
    state = torch.load(MODEL_DIR / "ES_2Channel_training_state.pt")
    best_val_loss = state['best_val_loss']
    epochs_no_improve = state['epochs_no_improve']

train_loss_sum = 0.0
train_steps = 0

# Train on sim data
model.train()
train_loader = DataLoader(TensorDataset(TM_train_list), batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g)

for (X_batch,) in train_loader:
    X_batch_raw = X_batch  # Already on GPU
    B = X_batch_raw.shape[0]
    X_in = torch.stack([X_batch_raw.real, X_batch_raw.imag], dim=1)
    # [B, 2, 42025, 145]

    X_batch = X_in.reshape(B, 2, 205, 205, 145)    # pixels -> spatial
    X_batch = X_batch.permute(0, 1, 4, 2, 3)        # -> [B, 2, 145, 205, 205]
    X_batch = F.pad(X_batch, (0, 3, 0, 3))          # -> [B, 2, 145, 208, 208]

    optimizer.zero_grad(set_to_none=True)
    pred, latent = model(X_batch)

    pred = pred[:, :, :, :205, :205]                 # [B, 2, 145, 205, 205]
    pred = pred.permute(0, 1, 3, 4, 2)              # [B, 2, 205, 205, 145]
    pred = pred.reshape(B, 2, 42025, 145)            # [B, 2, 42025, 145]
    pred_flat = pred.permute(0, 2, 3, 1)             # [B, 42025, 145, 2]
    pred_complex = torch.view_as_complex(pred_flat.contiguous())  # [B, 42025, 145]

    tr_weight_loss, tr_phase_loss, tr_amp_loss = phase_aligned_loss_wholeTM_perSample(pred_complex, X_batch_raw)
    train_loss_sum += tr_weight_loss.item()

    tr_weight_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    train_steps += 1

    del X_batch, X_batch_raw, pred, pred_complex, latent

# Validate on real exp data (no gradient updates - this is what drives early stopping)
model.eval()
exp_val_loader = DataLoader(TensorDataset(TM_exp_val), batch_size=exp_val_batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)

val_loss_sum = 0.0
val_phase_loss_sum = 0.0
val_amp_loss_sum = 0.0
val_steps = 0

with torch.no_grad():
    for (X_batch,) in exp_val_loader:
        X_batch_raw = X_batch  # Already on GPU
        B = X_batch_raw.shape[0]
        X_in = torch.stack([X_batch_raw.real, X_batch_raw.imag], dim=1)
        # [B, 2, 42025, 145]

        X_batch = X_in.reshape(B, 2, 205, 205, 145)    # pixels -> spatial
        X_batch = X_batch.permute(0, 1, 4, 2, 3)        # -> [B, 2, 145, 205, 205]
        X_batch = F.pad(X_batch, (0, 3, 0, 3))          # -> [B, 2, 145, 208, 208]

        pred, latent = model(X_batch)

        pred = pred[:, :, :, :205, :205]                 # [B, 2, 145, 205, 205]
        pred = pred.permute(0, 1, 3, 4, 2)              # [B, 2, 205, 205, 145]
        pred = pred.reshape(B, 2, 42025, 145)            # [B, 2, 42025, 145]
        pred_flat = pred.permute(0, 2, 3, 1)             # [B, 42025, 145, 2]
        pred_complex = torch.view_as_complex(pred_flat.contiguous())  # [B, 42025, 145]

        val_weight_loss, val_phase_loss, val_amp_loss = phase_aligned_loss_wholeTM_perSample(pred_complex, X_batch_raw)
        val_loss_sum += val_weight_loss.item()
        val_phase_loss_sum += val_phase_loss.item()
        val_amp_loss_sum += val_amp_loss.item()
        val_steps += 1

        del X_batch, X_batch_raw, pred, pred_complex, latent

# Clear memory
del TM_train_list, TM_exp_val, train_loader, exp_val_loader
print(f"Sub-batch {sub_batch} completed. Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")

avg_train_loss = train_loss_sum / train_steps
avg_val_loss = val_loss_sum / val_steps
avg_val_phase_loss = val_phase_loss_sum / val_steps
avg_val_amp_loss = val_amp_loss_sum / val_steps
print(f"Epoch {epoch}, Sub-batch {sub_batch}: Train (sim)={avg_train_loss:.3f}; "
      f"Val (exp) weight={avg_val_loss:.3f}, phase={avg_val_phase_loss:.3f}, amp={avg_val_amp_loss:.3f}")

state = {
    'best_val_loss': best_val_loss,
    'epochs_no_improve': epochs_no_improve,
    'epoch': epoch,
    'sub_batch': sub_batch
}

with open(MODEL_DIR / "ES_2Channel_loss_log.csv", 'a') as f:
    f.write(f"{epoch},{sub_batch},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")

with open(MODEL_DIR / "ES_2Channel_epoch_val_losses.txt", 'a') as f:
    f.write(f"{avg_val_loss:.6f}\n")

torch.save(model.state_dict(), MODEL_DIR / "ES_2Channel_int_weights.pt")
torch.save(optimizer.state_dict(), MODEL_DIR / "ES_2Channel_optimiser.pt")
torch.save(state, MODEL_DIR / "ES_2Channel_training_state.pt")
