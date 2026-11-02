import subprocess
import os
import shutil
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import paths

seed_value = 44
tm_dataset_size = 2000

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
MODEL_DIR   = str(paths.OUT_ROOT / "5m_fibre" / "62x_compression" / f"seed{seed_value}" / str(tm_dataset_size))
STATE_PATH  = os.path.join(MODEL_DIR, "ES_2Channel_training_state.pt")
INT_WEIGHTS = os.path.join(MODEL_DIR, "ES_2Channel_int_weights.pt")
BEST_WEIGHTS= os.path.join(MODEL_DIR, "ES_2Channel_best_weights.pt")
VAL_LOSSES  = os.path.join(MODEL_DIR, "ES_2Channel_epoch_val_losses.txt")
LOSS_LOG    = os.path.join(MODEL_DIR, "ES_2Channel_loss_log.csv")

# ── Hyperparameters ───────────────────────────────────────────────────────────
START_EPOCH       = 0  # Set to 0 to start fresh, or >0 to resume from that epoch
NUM_EPOCHS        = 1000
NUM_SUB_BATCHES   = 20       # tm_dataset_size / 100 per sub-batch - update alongside tm_dataset_size above
PRETRAIN_PATIENCE = 30

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Initialise training state if starting fresh ───────────────────────────────
if not os.path.exists(STATE_PATH):
    torch.save({'best_val_loss': float('inf'), 'epochs_no_improve': 0}, STATE_PATH)
    print("Initialised fresh training state")
else:
    print("Resuming from existing training state")

# ── Initialise loss log with header if starting fresh ────────────────────────
if not os.path.exists(LOSS_LOG):
    with open(LOSS_LOG, 'w') as f:
        f.write("epoch,avg_val_loss\n")

# ── Main training loop ────────────────────────────────────────────────────────
for epoch in range(START_EPOCH, NUM_EPOCHS):

    # Clear sub-batch val loss accumulator for this epoch
    open(VAL_LOSSES, 'w').close()

    # Run all sub-batches for this epoch
    for sub_batch in range(NUM_SUB_BATCHES):
        print(f"\n── Epoch {epoch} | Sub-batch {sub_batch + 1}/{NUM_SUB_BATCHES} ──")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "ES_2Channel_Autoencoder.py"), str(epoch), str(sub_batch)],
            cwd=SCRIPT_DIR,
            check=False
        )
        if result.returncode != 0:
            print(f"ERROR: subprocess failed at epoch {epoch}, sub_batch {sub_batch}. Stopping.")
            raise RuntimeError(f"Subprocess failed at epoch {epoch}, sub_batch {sub_batch}")

    # All sub-batches done — average val losses across sub-batches
    with open(VAL_LOSSES, 'r') as f:
        sub_losses = [float(line.strip()) for line in f if line.strip()]

    if len(sub_losses) != NUM_SUB_BATCHES:
        print(f"WARNING: expected {NUM_SUB_BATCHES} val losses, got {len(sub_losses)}. Skipping early stopping for this epoch.")
        continue

    epoch_val_loss = sum(sub_losses) / len(sub_losses)
    print(f"\nEpoch {epoch} complete — avg exp val loss: {epoch_val_loss:.4f} (sub-batch losses: {[round(l,4) for l in sub_losses]})")

    # Append to loss log
    with open(LOSS_LOG, 'a') as f:
        f.write(f"{epoch},{epoch_val_loss:.6f}\n")

    # Early stopping check (on real exp val loss)
    state = torch.load(STATE_PATH)

    if epoch_val_loss < state['best_val_loss']:
        print(f"  Val loss improved: {state['best_val_loss']:.4f} -> {epoch_val_loss:.4f}. Saving best weights.")
        state['best_val_loss']    = epoch_val_loss
        state['epochs_no_improve'] = 0
        shutil.copy(INT_WEIGHTS, BEST_WEIGHTS)
    else:
        state['epochs_no_improve'] += 1
        print(f"  No improvement. Patience: {state['epochs_no_improve']}/{PRETRAIN_PATIENCE}")
        if state['epochs_no_improve'] >= PRETRAIN_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            torch.save(state, STATE_PATH)
            break

    torch.save(state, STATE_PATH)

print("\nSim-training / exp-early-stopped run complete.")
print(f"Best exp val loss: {torch.load(STATE_PATH)['best_val_loss']:.4f}")
print(f"Best weights saved to: {BEST_WEIGHTS}")
