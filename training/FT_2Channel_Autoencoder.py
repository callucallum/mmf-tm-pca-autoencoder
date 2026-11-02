from datetime import datetime
import os
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths
from autoencoder_functions import (
    TM_Autoencoder,
    phase_aligned_loss_wholeTM_perSample,
)

finetune_lr = 0.000045

seed_value = 44
tm_dataset_size = 2000

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

DATASPLIT_DIR = paths.EXP_SPLIT_DIR
MODEL_DIR = paths.OUT_ROOT / "5m_fibre" / "124x_compression" / f"seed{seed_value}" / f"{tm_dataset_size}"

train_idx = np.load(DATASPLIT_DIR / "train_split_idx.npy")
val_idx = np.load(DATASPLIT_DIR / "val_split_idx.npy")

dataset = np.load(paths.EXP_DATA, allow_pickle=True, mmap_mode="r")

TM_stack = dataset['TM']

TM_train = torch.tensor(TM_stack[train_idx], dtype=torch.cfloat, device=device)
TM_val   = torch.tensor(TM_stack[val_idx], dtype=torch.cfloat, device=device)

print(f"Found {len(TM_stack)} experimental files, train: {len(TM_train)}, val: {len(TM_val)}")

batch_size=10
best_val_loss = float("inf")
best_model_state = None
epochs_no_improve = 0

epoch = int(sys.argv[1])
sub_batch = int(sys.argv[2])

if epoch == 0 and sub_batch == 0:
    model = TM_Autoencoder(latent_dim=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=1.108781e-06)
    model.load_state_dict(torch.load(MODEL_DIR / "PT_TMM_2Channel_best_weights_Hadamard.pt"))  # load best pretrained
    optimizer.load_state_dict(torch.load(MODEL_DIR / "PT_TMM_2Channel_optimiser_Hadamard.pt"))
else:
    model = TM_Autoencoder(latent_dim=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=1.108781e-06)
    model.load_state_dict(torch.load(MODEL_DIR / "FT_2Channel_int_weights.pt"))
    optimizer.load_state_dict(torch.load(MODEL_DIR / "FT_2Channel_optimiser.pt"))

if epoch > 0 or sub_batch > 0:
    state = torch.load(MODEL_DIR / "FT_2Channel_training_state.pt")
    best_val_loss = state['best_val_loss']
    epochs_no_improve = state['epochs_no_improve']

train_loss_sum = 0.0
train_steps = 0
val_loss_sum = 0.0
val_steps = 0

# Train
model.train()
train_loader = DataLoader(TensorDataset(TM_train), batch_size=2, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g)

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
    
    weight_loss, _, _1 = phase_aligned_loss_wholeTM_perSample(pred_complex, X_batch_raw)
    train_loss_sum += weight_loss.item()
    
    weight_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    train_steps += 1
    
    del X_batch, X_batch_raw, pred, pred_complex, latent

# Validation
model.eval()
val_loader = DataLoader(TensorDataset(TM_val), batch_size=2, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)

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
        
        val_phase_loss, _, _1 = phase_aligned_loss_wholeTM_perSample(pred_complex, X_batch_raw)
        val_loss_sum += val_phase_loss.item()
        val_steps += 1
        
        del X_batch, X_batch_raw, pred, pred_complex, latent

# Clear memory
del TM_train, TM_val, train_loader, val_loader
print(f"Sub-batch {sub_batch} completed. Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")

# Sub-batch summary
avg_train_loss = train_loss_sum / train_steps
avg_val_loss = val_loss_sum / val_steps
print(f"Epoch {epoch}, Sub-batch {sub_batch}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

state = {
    'best_val_loss': best_val_loss,
    'epochs_no_improve': epochs_no_improve,
    'epoch': epoch,
    'sub_batch': sub_batch
}

with open(MODEL_DIR / "FT_2Channel_loss_log.csv", 'a') as f:
    f.write(f"{epoch},{sub_batch},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")

with open(MODEL_DIR / "FT_2Channel_epoch_val_losses.txt", 'a') as f:
    f.write(f"{avg_val_loss:.6f}\n")

torch.save(model.state_dict(), MODEL_DIR / "FT_2Channel_int_weights.pt")
torch.save(optimizer.state_dict(), MODEL_DIR / "FT_2Channel_optimiser.pt")

# Add this before the end:
torch.save(state, MODEL_DIR / "FT_2Channel_training_state.pt")