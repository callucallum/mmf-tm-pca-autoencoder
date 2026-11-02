import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths
from autoencoder_functions import TM_Autoencoder

# Downstream "unscrambling" test: use a RECONSTRUCTED TM the way an endoscope would, and judge it on the TRUE TM.
# The TM maps 145 random orthonormal input patterns to a 205x205 output field, so only a 145-dimensional subspace of
# inputs can be driven. Two tasks, each compared with the same procedure run on the true TM (the ideal for 145 inputs):
#   1. Focusing: for random output pixels, set the input by phase conjugation with the reconstructed TM, apply it to the
#      true TM, and measure the focus enhancement (peak intensity / mean intensity) as a fraction of the ideal.
#   2. Target image: find the input that best produces the target intensity with the reconstructed TM (Gerchberg-Saxton
#      with a least-squares step), apply it to the true TM, and compare the output intensity with the target
#      (correlation and SSIM).
# Models: AE sim-pretrained (ES weights), AE fine-tuned on each --ft-n, and PCA fitted on exp training TMs (62x).
# Scored on the exp VALIDATION TMs only. Usage:
#   python evaluation/unscramble_test.py [--pattern F|ring|spots | --target path/to/image.png] [--ae-dir <weights>/final_62x/trial16/seed42] [--ft-n 10 163]
# Weights are read from <TM_MODEL_ROOT>/final_62x/trial16/seed42 by default (the weights/ folder of this repository); results go to <TM_OUT_ROOT>/unscramble_test.
parser = argparse.ArgumentParser()
parser.add_argument("--target", default="", help="target image (png/jpg/npy); overrides --pattern")
parser.add_argument("--pattern", default="spots", choices=["spots", "ring", "F"], help="built-in target when --target is not given")
parser.add_argument("--pca-sim", type=int, default=163, help="also fit PCA on this many simulated composite-5m TMs (0 exp TMs); 0 to skip")
parser.add_argument("--shift", type=int, default=0, help="horizontal shift (px) applied to the validation TMs before any model sees them")
parser.add_argument("--shift-mode", default="crop", choices=["crop", "roll"],
                    help="crop: pixels shifted past the edge are dropped and the vacated strip is filled (see --fill-level); roll: circular wrap (earlier runs)")
parser.add_argument("--fill-level", type=float, default=0.05,
                    help="crop mode: RMS of the complex Gaussian noise in the vacated strip, relative to each TM's overall RMS (0 = zeros)")
parser.add_argument("--ae-dir", default="", help="folder with ES_best_weights.pt, FT_best_weights_N*.pt, hyperparameters.json (default: <TM_MODEL_ROOT>/final_62x/trial16/seed42)")
parser.add_argument("--ft-n", type=int, nargs="+", default=[10, 163], help="fine-tuned autoencoder weights to include")
parser.add_argument("--split", choices=["val", "test"], default="val",
                    help="which split is scored. test: the held-out split; run it ONCE, after every model and table is frozen")
parser.add_argument("--extra-ae", nargs="+", default=[],
                    help='extra autoencoders as "label=file", file inside --ae-dir, e.g. "AE specialist, 2 exp TMs=FewShot_exp_N2.pt"')
parser.add_argument("--pca-exp-n", type=int, default=163, help="number of original exp train TMs for the exp-fitted PCA (163 = all; use e.g. 2 for a few-TM comparison)")
parser.add_argument("--no-pca", action="store_true", help="skip the PCA comparison (it needs a ~minute fit and ~20 GB of GPU memory)")
parser.add_argument("--pca-k", type=int, default=1352, help="PCA components (1352 = 62x)")
parser.add_argument("--n-focus", type=int, default=300, help="number of random focus pixels per TM")
parser.add_argument("--gs-iters", type=int, default=30)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

PIXELS, GRID, N_IN = 42025, 205, 145
EVAL_CHUNK = 4
device = torch.device(args.device)

EXP_SPLIT_DIR = paths.EXP_SPLIT_DIR
AE_DIR = Path(args.ae_dir) if args.ae_dir else paths.MODEL_ROOT / "final_62x" / "trial16" / "seed42"
OUT_DIR = paths.OUT_ROOT / "unscramble_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEM = (Path(args.target).stem if args.target else args.pattern) + \
       ((f"_shift{args.shift}" if args.shift_mode == "roll" else f"_shiftcrop{args.shift}") if args.shift else "") + \
       (f"_{args.split}" if args.split != "val" else "")

exp_stack = paths.load_exp_stack()


def load_tms(idx):
    return torch.tensor(exp_stack[idx], dtype=torch.cfloat)      # [N, 42025, 145]


tm_val = load_tms(np.load(EXP_SPLIT_DIR / f"{args.split}_split_idx.npy"))   # name kept from the val-only version; holds the chosen split
print(f"{len(tm_val)} {args.split} TMs; device {device}; autoencoder weights from {AE_DIR}")


# ── target image placed inside the bright disc of the output field ────────────
def load_target(path):
    if not path:
        img = np.zeros((64, 64), np.float32)
        yy, xx = np.mgrid[:64, :64]
        if args.pattern == "F":
            img[8:56, 12:22] = 1; img[8:18, 12:52] = 1; img[28:36, 12:42] = 1
        elif args.pattern == "ring":
            r = np.hypot(yy - 32, xx - 32)
            img[(r > 18) & (r < 28)] = 1
        else:                                                 # three bright spots on a triangle
            for cy, cx in ((16, 32), (46, 16), (46, 48)):
                img[np.hypot(yy - cy, xx - cx) < 5.5] = 1
        return img
    if Path(path).suffix == ".npy":
        img = np.load(path).astype(np.float32)
    else:
        from PIL import Image
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return img / max(img.max(), 1e-8)


def place_target(img, intensity_map):
    """Resize img to fit inside the bright disc of the output field and paste it at the centre of a 205x205 canvas."""
    support = intensity_map > 0.2 * intensity_map.max()
    rows, cols = np.where(support.any(1))[0], np.where(support.any(0))[0]
    side = int(min(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1) / np.sqrt(2) * 0.95)
    cy, cx = (rows.min() + rows.max()) // 2, (cols.min() + cols.max()) // 2
    t = F.interpolate(torch.from_numpy(img)[None, None], size=(side, side), mode="bilinear", antialias=True, align_corners=False)[0, 0]
    canvas = torch.zeros(GRID, GRID)
    y0, x0 = cy - side // 2, cx - side // 2
    canvas[y0:y0 + side, x0:x0 + side] = t.clamp(min=0)
    return (canvas / canvas.max()).numpy(), support


def shift_tm(tm, s):
    """Move the 205x205 output image of every input by s columns. roll: circular. crop: pixels past the edge are lost and the
    vacated strip gets complex Gaussian noise (RMS = fill-level x the TM's overall RMS), like a camera that moved."""
    B = len(tm)
    t = tm.reshape(B, GRID, GRID, N_IN)
    if args.shift_mode == "roll":
        return t.roll(s, dims=2).reshape(B, PIXELS, N_IN)
    out = torch.zeros_like(t)
    if s > 0:
        out[:, :, s:] = t[:, :, :GRID - s]
        strip = (slice(None), slice(None), slice(0, s))
    else:
        out[:, :, :GRID + s] = t[:, :, -s:]
        strip = (slice(None), slice(None), slice(GRID + s, GRID))
    if args.fill_level > 0:
        rms = t.abs().pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()
        g = torch.Generator().manual_seed(0)                      # same noise for every model
        shape = out[strip].shape
        noise = torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g)) / 2 ** 0.5
        out[strip] = args.fill_level * rms * noise
    return out.reshape(B, PIXELS, N_IN)


if args.shift:                                                   # the TMs a shifted camera would measure; models and the "true TM" all see these
    tm_val = shift_tm(tm_val, args.shift)
# Computed AFTER the shift, so the target and the focus pixels sit on the shifted bright disc (before, focus pixels that
# moved into the dark margin had zero rows -> NaN for the ideal and zero focus for every model).
intensity_map = tm_val.abs().pow(2).mean(dim=(0, 2)).reshape(GRID, GRID).numpy()
target, support = place_target(load_target(args.target), intensity_map)
target_t = torch.from_numpy(target).to(device)
amp = target_t.reshape(-1).sqrt()                                # target amplitude per output pixel

gen = torch.Generator(device=device).manual_seed(0)
phase0 = torch.rand(PIXELS, device=device, generator=gen) * 2 * np.pi   # same starting phase for every model and TM
bright = torch.from_numpy(np.where(intensity_map.reshape(-1) > 0.25 * intensity_map.max())[0]).to(device)
pix = bright[torch.randperm(len(bright), device=device, generator=gen)[:args.n_focus]]   # same focus pixels everywhere


# ── metrics ───────────────────────────────────────────────────────────────────
def focus_enhancement(T, That):
    """Phase-conjugation focus at each pixel in `pix` computed from That, applied to T: peak / mean intensity."""
    rows = That[pix]
    C = rows.conj() / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)   # [m, 145]
    Y = T @ C.T                                                  # [P, m] output fields on the true TM
    peak = Y[pix, torch.arange(len(pix), device=device)].abs().pow(2)
    bg = Y.abs().pow(2).mean(0)
    # A model whose TM is exactly zero at a focus pixel (PCA fitted on exp TMs has zero rows outside the training disc) cannot
    # drive any light there: its input is all zeros, the enhancement is 0, not 0/0.
    return torch.where(bg > 0, peak / bg.clamp_min(1e-30), torch.zeros_like(peak))


def project_image(That):
    """Gerchberg-Saxton with a least-squares step on That; returns the 145 input coefficients."""
    G = That.conj().T @ That
    G = G + 1e-6 * G.diagonal().real.mean() * torch.eye(N_IN, device=device)
    y = amp * torch.exp(1j * phase0)
    for _ in range(args.gs_iters):
        c = torch.linalg.solve(G, That.conj().T @ y)
        y = amp * torch.exp(1j * (That @ c).angle())
    return c


def gaussian_window(size=11, sigma=1.5):
    g = torch.exp(-0.5 * ((torch.arange(size, dtype=torch.float32) - size // 2) / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].to(device)


WINDOW = gaussian_window()


def ssim(a, b):
    """SSIM of two [H, W] images already scaled to [0, 1]."""
    a, b = a[None, None], b[None, None]
    mu_a, mu_b = F.conv2d(a, WINDOW), F.conv2d(b, WINDOW)
    var_a = F.conv2d(a * a, WINDOW) - mu_a ** 2
    var_b = F.conv2d(b * b, WINDOW) - mu_b ** 2
    cov = F.conv2d(a * b, WINDOW) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean().item()


def image_scores(T, That):
    c = project_image(That)
    out = (T @ c).abs().pow(2).reshape(GRID, GRID)
    out = out / out.max()
    corr = torch.corrcoef(torch.stack([out.reshape(-1), target_t.reshape(-1)]))[0, 1].item()
    return corr, ssim(out, target_t), out


# ── reconstructions ───────────────────────────────────────────────────────────
@torch.no_grad()
def ae_reconstruct(model, X):
    B = X.shape[0]
    X_in = torch.stack([X.real, X.imag], dim=1).reshape(B, 2, GRID, GRID, N_IN).permute(0, 1, 4, 2, 3)
    pred, _ = model(F.pad(X_in, (0, 3, 0, 3)), show_shapes=False)
    pred = pred[:, :, :, :GRID, :GRID].permute(0, 1, 3, 4, 2).reshape(B, 2, PIXELS, N_IN)
    return torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())


def to_patterns(tm):
    p = tm.permute(0, 2, 1).reshape(-1, PIXELS)
    return torch.cat([p.real, p.imag], dim=1).float()


def to_tm(patterns, n_tm):
    c = torch.complex(patterns[:, :PIXELS], patterns[:, PIXELS:])
    return c.reshape(n_tm, N_IN, PIXELS).permute(0, 2, 1)


def sim_patterns(n_tm, seed=0):
    """Speckle patterns of n_tm simulated composite-5m TMs (same forward model and noise as the autoencoder pretraining;
    a copy of the function in pca_baseline.py)."""
    import pickle
    sim_dir = paths.SIM_DIR
    sim_idx = np.load(paths.SIM_SPLIT_DIR / "train_composite5m_idx_2000.npy")[:n_tm]
    unique_ids = np.unique(sim_idx.ravel()).astype(np.int64)
    pos_of = {int(i): n for n, i in enumerate(unique_ids)}
    rows = np.array([[pos_of[int(i)] for i in row] for row in sim_idx])
    pool = []
    for i in unique_ids:
        stem = f"TM{int(i):04d}"
        path = next((p for p in (sim_dir / f"{stem}.pkl", sim_dir / f"{stem}-GPU.pkl") if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No pickle found for {stem}")
        with open(path, "rb") as f:
            pool.append(torch.from_numpy(pickle.load(f)["tm_mode"]).to(torch.complex64))
    pool = torch.stack(pool).to(device)
    with open(sim_dir / "dist_pim_to_real.pkl", "rb") as f:
        dist_pim_to_real = torch.tensor(pickle.load(f)).to(device=device, dtype=torch.complex64)
    with open(sim_dir / "prox_real_to_pim.pkl", "rb") as f:
        prox_real_to_pim = torch.from_numpy(pickle.load(f)).to(device=device, dtype=torch.complex64)
    u_in = torch.tensor(np.load(paths.INPUTS_FILE),
                        dtype=torch.complex64, device=device)
    a_in = prox_real_to_pim @ u_in
    a_in_i = a_in / (torch.linalg.vector_norm(a_in, dim=0, keepdim=True) + 1e-8)
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for s in range(0, n_tm, 4):
        r = rows[s:s + 4]
        T = pool[r[:, 0]]
        for j in range(1, 5):
            T = pool[r[:, j]] @ T
        X = dist_pim_to_real @ (T @ a_in_i)
        B = X.shape[0]
        amp_noise = torch.empty(B, 1, 1, device=device).uniform_(0.01, 0.15, generator=g)
        phase_noise = torch.empty(B, 1, 1, device=device).uniform_(0.01, 0.1, generator=g)
        amp, phase = X.abs(), X.angle()
        amp = amp + amp_noise * amp * torch.randn(amp.shape, device=device, generator=g)
        phase = phase + phase_noise * torch.randn(phase.shape, device=device, generator=g)
        X = amp * torch.exp(1j * phase)
        X = X / (torch.linalg.svdvals(X)[:, 0][:, None, None] + 1e-8)
        out.append(to_patterns(X.cpu()))
    return torch.cat(out)


def fit_pca(X, k):
    """X: [n_patterns, 84050] real training patterns."""
    X = X.to(device)
    mean = X.mean(0, keepdim=True)
    X -= mean
    w, U = torch.linalg.eigh((X @ X.T).double())
    w, U = w.flip(0)[:k], U.flip(1)[:, :k]
    keep = w > 1e-6 * w[0]                                       # few TMs: fewer than k real directions; drop the numerical-noise ones
    U = U[:, keep]
    V = X.T @ U.float()
    V /= V.norm(dim=0, keepdim=True)                             # unit-norm principal directions, [84050, k]
    return mean, V


recon = {}   # name -> function(X [B,P,145] on device) -> reconstructed TMs
hp = json.loads((AE_DIR / "hyperparameters.json").read_text())["params"]


def load_ae(file):
    m = TM_Autoencoder(enc_channels=[hp["enc1"], 2 * hp["enc1"], 3 * hp["enc1"]], latent_dim=8,
                       attn_heads=hp["attn_heads"], linformer_k=hp["linformer_k"]).to(device)
    m.load_state_dict(torch.load(AE_DIR / file, map_location=device))
    m.eval()
    return lambda X, m=m: ae_reconstruct(m, X)


recon["AE sim-pretrained"] = load_ae("ES_best_weights.pt")
for n in args.ft_n:
    recon[f"AE fine-tuned on {n} exp TMs"] = load_ae(f"FT_best_weights_N{n}.pt")
for item in args.extra_ae:
    label, file = item.rsplit("=", 1)
    recon[label] = load_ae(file)
if not args.no_pca:
    def pca_recon(mean, V):
        return lambda X: to_tm((to_patterns(X.cpu()).to(device) - mean) @ V @ V.T + mean, len(X))

    print(f"Fitting PCA on {args.pca_exp_n} exp training TMs ...")
    recon[f"PCA 62x ({args.pca_exp_n} exp TMs)"] = pca_recon(*fit_pca(to_patterns(load_tms(np.load(EXP_SPLIT_DIR / "train_split_idx.npy")[:args.pca_exp_n])), args.pca_k))
    if args.pca_sim:
        print(f"Simulating {args.pca_sim} composite-5m TMs and fitting PCA on them ...")
        recon[f"PCA 62x ({args.pca_sim} sim TMs)"] = pca_recon(*fit_pca(sim_patterns(args.pca_sim), args.pca_k))

# ── run ───────────────────────────────────────────────────────────────────────
rows, gallery = [], {"target": target}
for i in range(0, len(tm_val), EVAL_CHUNK):
    X = tm_val[i:i + EVAL_CHUNK].to(device)
    recs = {name: f(X) for name, f in recon.items()}
    for j in range(len(X)):
        T = X[j]
        eta_ideal = focus_enhancement(T, T)
        corr, s, out = image_scores(T, T)
        rows.append(("ideal (true TM)", i + j, eta_ideal.mean().item(), 1.0, corr, s))
        if i + j == 0:
            gallery["ideal (true TM)"] = out.cpu().numpy()
        for name, R in recs.items():
            eta = focus_enhancement(T, R[j])
            corr, s, out = image_scores(T, R[j])
            rows.append((name, i + j, eta.mean().item(), (eta / eta_ideal).mean().item(), corr, s))
            if i + j == 0:
                gallery[name] = out.cpu().numpy()
    print(f"  done {min(i + EVAL_CHUNK, len(tm_val))}/{len(tm_val)} TMs", end="\r")
print()

with open(OUT_DIR / f"unscramble_results_{STEM}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "tm", "focus_enhancement", "focus_fraction_of_ideal", "image_corr", "image_ssim"])
    w.writerows([(m, t, round(e, 3), round(fr, 4), round(c, 4), round(s, 4)) for m, t, e, fr, c, s in rows])
np.savez(OUT_DIR / f"unscramble_gallery_{STEM}.npz", **gallery)

print(f"\nTarget: {STEM}; focus pixels per TM: {args.n_focus}; GS iterations: {args.gs_iters}; shift: {args.shift} px" + (f" ({args.shift_mode}, fill {args.fill_level})" if args.shift else ""))
print(f"{'model':36s} {'focus enh.':>11s} {'% of ideal':>11s} {'img corr':>9s} {'img SSIM':>9s}")
for name in ["ideal (true TM)"] + list(recon):
    r = np.array([x[2:] for x in rows if x[0] == name], dtype=float)
    print(f"{name:36s} {r[:, 0].mean():11.1f} {100 * r[:, 1].mean():10.1f}% {r[:, 2].mean():9.3f} {r[:, 3].mean():9.3f}")
print(f"\nSaved to {OUT_DIR}")
