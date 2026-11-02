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
tm_dataset_size = 3000

DATA_DIR = paths.SIM_DIR
MODEL_DIR = paths.OUT_ROOT / "5m_fibre" / "124x_compression" / f"seed{seed_value}" / f"{tm_dataset_size}"

rng = np.random.default_rng(seed_value)
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # for GPU
torch.backends.cudnn.deterministic = True  # ensure deterministic operations
torch.backends.cudnn.benchmark = False
#torch.utils.deterministic.fill_uninitialized_memory = True # deactivated as of 17/03 as it's detremental to performance, but can be used if reproducibility become a real issue (see: https://pytorch.org/docs/stable/notes/randomness.html)

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

# directory containing the files
directory = DATA_DIR

train_split_idx = np.load(paths.SIM_SPLIT_DIR / f"train_composite5m_idx_{tm_dataset_size}.npy")
val_split_idx = np.load(paths.SIM_SPLIT_DIR / "val_composite5m_idx.npy")

for split_name, split_indices in (("training", train_split_idx), ("validation", val_split_idx)):
    if split_indices.ndim != 2 or split_indices.shape[1] != 5:
        raise ValueError(
            f"Expected {split_name} indices with shape [N, 5], got {split_indices.shape}"
        )

batch_size_data = 100
num_sub_batches = math.ceil(len(train_split_idx) / batch_size_data)

start_idx = sub_batch * batch_size_data
end_idx = min((sub_batch + 1) * batch_size_data,
              len(train_split_idx))

train_idx_sub = train_split_idx[start_idx:end_idx]

required_tm_indices = np.unique(
    np.concatenate((train_idx_sub.ravel(), val_split_idx.ravel()))
).astype(np.int64)
required_tm_index_set = set(required_tm_indices.tolist())

print(
    f"Sub-batch {sub_batch}: "
    f"train_idx[{start_idx}:{end_idx}] "
    f"-> {len(train_idx_sub)} samples")

# Load only the files needed for this training sub-batch and validation.
tmm_tm_mode = {}

for dataset_idx in required_tm_index_set:
    filename_stem = f"TM{int(dataset_idx):04d}"
    candidate_paths = (
        directory / f"{filename_stem}.pkl",
        directory / f"{filename_stem}-GPU.pkl",
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

print(f"Loaded {len(tmm_tm_mode)}/{len(required_tm_index_set)} required files to GPU")

def build_composite_tm(row_indices, tm_pool):
    """row_indices: array of shape [5], tm_pool: full tmm_tm_mode tensor/list"""
    assert len(row_indices) == 5, (
        f"Expected 5 TM indices, got {len(row_indices)}"
    )
    T = tm_pool[row_indices[0]]
    for idx in row_indices[1:]:
        T = tm_pool[idx] @ T  # fix propagation order convention here
    return T

# Build composite TMs for this sub-batch: [100, 5] -> [100, H, W]
TM_train_sub = torch.stack([
    build_composite_tm(row, tmm_tm_mode) for row in train_idx_sub
])

# Same for val (assuming val_split_idx is also [N_val, 5] now)
TM_val_sub = torch.stack([
    build_composite_tm(row, tmm_tm_mode) for row in val_split_idx
])


with open(DATA_DIR / 'dist_pim_to_real.pkl', 'rb') as f:
    dist_pim_to_real = pickle.load(f)
    dist_pim_to_real = torch.tensor(dist_pim_to_real).to(device)
    print(f'dist_pim_to_real: {dist_pim_to_real.shape}')

with open(DATA_DIR / 'prox_real_to_pim.pkl', 'rb') as f:
    prox_real_to_pim = pickle.load(f)
    prox_real_to_pim = torch.from_numpy(prox_real_to_pim).to(device)
    print(f'prox_real_to_pim: {prox_real_to_pim.shape}')

n_points_modes_371 = 371
num_inputs = 145

# Same random inputs every time
u_in_2d = np.load(paths.INPUTS_FILE)

u_in = torch.tensor(u_in_2d, dtype=torch.complex64, device=device)

print(f'u_in shape: {u_in.shape}')


print(f"First file TM shape: {TM_train_sub[0].shape}")

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

TM_val_list = []
for tm_x in TM_val_sub:
    TM_val_list.append(calculate_forward_tm(tm_x))

TM_train_list = torch.stack(TM_train_list)
TM_val_list = torch.stack(TM_val_list)

del dist_pim_to_real, prox_real_to_pim

print(f"Found {len(TM_train_list)} training files, TM shape: {TM_train_list[0].shape}, no. of TMs: {len(TM_train_list)}")
print(f"Found {len(TM_val_list)} validation files, TM shape: {TM_val_list[0].shape}, no. of TMs: {len(TM_val_list)}")

print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

batch_size_data = 100  # Load 100 from each directory at a time

batch_size=8
best_val_loss = float("inf")
best_model_state = None
epochs_no_improve = 0
pretrain_lr = 0.001529

if epoch == 0 and sub_batch == 0:
    torch.manual_seed(seed_value)  # re-seed right before init so weight init isn't shifted by how much
    torch.cuda.manual_seed_all(seed_value)  # RNG-consuming data-loading work preceded it in this script
    model = TM_Autoencoder(latent_dim=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=1.108781e-06)
    print("Model initialised from scratch")
else:
    model = TM_Autoencoder(latent_dim=4).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay= 1.108781e-06)
    optimizer.load_state_dict(torch.load(MODEL_DIR / "PT_TMM_2Channel_optimiser_Hadamard.pt"))
    model.load_state_dict(torch.load(MODEL_DIR / "PT_TMM_2Channel_int_weights_Hadamard.pt"))
    print("Imported most recent model weights and optimiser")

if epoch > 0 or sub_batch > 0:
    state = torch.load(MODEL_DIR / "PT_TMM_2Channel_training_state_Hadamard.pt")
    best_val_loss = state['best_val_loss']
    epochs_no_improve = state['epochs_no_improve']

train_loss_sum = 0.0
train_steps = 0

# Train
model.train()
train_loader = DataLoader(TensorDataset(TM_train_list), batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g)

for (X_batch,) in train_loader:
    X_batch_raw = X_batch  # Already on GPU
    B = X_batch_raw.shape[0]
    X_in = torch.stack([X_batch_raw.real, X_batch_raw.imag], dim=1)  
    # [B, 2, 42025, 145]

    # Reshape pixels to spatial, THEN permute modes into position
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

# Validation
model.eval()
val_loader = DataLoader(TensorDataset(TM_val_list), batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)

val_loss_sum = 0.0
val_phase_loss_sum = 0.0
val_amp_loss_sum = 0.0
val_steps = 0

with torch.no_grad():
    for (X_batch,) in val_loader:
        X_batch_raw = X_batch  # Already on GPU
        B = X_batch_raw.shape[0]
        X_in = torch.stack([X_batch_raw.real, X_batch_raw.imag], dim=1)  
        # [B, 2, 42025, 145]

        # Reshape pixels to spatial, THEN permute modes into position
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
del TM_train_list, TM_val_list, train_loader, val_loader
print(f"Sub-batch {sub_batch} completed. Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")

# Sub-batch summary
avg_train_loss = train_loss_sum / train_steps
avg_val_loss = val_loss_sum / val_steps
avg_val_phase_loss = val_phase_loss_sum / val_steps
avg_val_amp_loss = val_amp_loss_sum / val_steps
print(f"Epoch {epoch}, Sub-batch {sub_batch}: Train={avg_train_loss:.3f}; Val weight={avg_val_loss:.3f}, phase={avg_val_phase_loss:.3f}, amp={avg_val_amp_loss:.3f}")

state = {
    'best_val_loss': best_val_loss,
    'epochs_no_improve': epochs_no_improve,
    'epoch': epoch,
    'sub_batch': sub_batch
}

with open(MODEL_DIR / "PT_TMM_2Channel_loss_log_Hadamard.csv", 'a') as f:
    f.write(f"{epoch},{sub_batch},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")

with open(MODEL_DIR / "PT_TMM_2Channel_epoch_val_losses_Hadamard.txt", 'a') as f:
    f.write(f"{avg_val_loss:.6f}\n")

torch.save(model.state_dict(), MODEL_DIR / "PT_TMM_2Channel_int_weights_Hadamard.pt")
torch.save(optimizer.state_dict(), MODEL_DIR / "PT_TMM_2Channel_optimiser_Hadamard.pt")

# Add this before the end:
torch.save(state, MODEL_DIR / "PT_TMM_2Channel_training_state_Hadamard.pt")