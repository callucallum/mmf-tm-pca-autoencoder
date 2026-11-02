import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths
from autoencoder_functions import phase_aligned_loss_wholeTM_perSample

# Per-speckle PCA baseline at the autoencoder's compression ratios.
# Each of the 145 inputs of a TM is one sample: a 205x205 complex speckle pattern = 84,050 reals.
# The autoencoder compresses one such pattern to latent_dim x 13 x 13 reals, so PCA keeps latent_dim * 169 components.
# Fit on exp training TMs only (--fit-n), scored with the same phase-aligned whole-TM error as the autoencoder.
# Usage: python pca_baseline.py --fit-n 10 163 [--eval-test] [--device cuda]
parser = argparse.ArgumentParser()
parser.add_argument("--fit-n", type=int, nargs="+", default=[163], help="number of exp training TMs to fit PCA on (10, 25, ..., 163)")
parser.add_argument("--ratios", type=int, nargs="+", default=[16, 31, 62, 83, 124], help="compression ratios (16/31/62/83/124 = latent_dim 32/16/8/6/4)")
parser.add_argument("--sim-val", type=int, default=20,
                    help="with --fit-sim: also score the sim-fitted PCA on this many HELD-OUT simulated composite-5m validation TMs (split 'simval'); 0 to skip")
parser.add_argument("--exp-on-sim", action="store_true",
                    help="also score every exp-fitted PCA on the held-out simulated validation TMs (the reverse domain gap); each exp fit then loads the simulator once more")
parser.add_argument("--fit-sim", type=int, default=0, help="also fit PCA on this many simulated composite-5m TMs (zero exp TMs); ~100 is plenty. Use --fit-n 0 to skip the exp fits")
parser.add_argument("--shift-test", action="store_true", help="score PCA and a saved 62x autoencoder on exp val TMs shifted by --shifts pixels -> pca_shift_test.csv")
parser.add_argument("--shifts", type=int, nargs="+", default=[0, 3, 5, 6, 8, 12, 20], help="horizontal pixel shifts (np.roll) for --shift-test")
parser.add_argument("--shift-mode", default="crop", choices=["crop", "roll"],
                    help="shift perturbation: crop = pixels past the edge are dropped and the vacated strip is filled with noise (see --fill-level); roll = circular wrap (earlier runs)")
parser.add_argument("--fill-level", type=float, default=0.05,
                    help="crop mode: RMS of the complex Gaussian noise in the vacated strip, relative to each TM's overall RMS (0 = zeros)")
parser.add_argument("--perturb", nargs="+", default=["shift"], choices=["shift", "rotate", "scale", "blur", "noise"],
                    help="perturbations for --shift-test (values fixed in PERTURB_VALUES: shift px, rotate deg, scale factor, blur sigma px, noise rel. to RMS)")
parser.add_argument("--ae-dir", default="", help="folder with ES_best_weights.pt / FT_best_weights_N*.pt / hyperparameters.json (default: <TM_MODEL_ROOT>/final_62x/trial16/seed42)")
parser.add_argument("--ae-ft-n", type=int, nargs="+", default=[10], help="fine-tuned autoencoder weights to include (5, 10, 163)")
parser.add_argument("--rob-ae", nargs="+", default=[], help='--shift-test: extra autoencoders as "label=file" (file inside --ae-dir), e.g. "AE specialist, 2 exp TMs=FewShot_exp_N2.pt"')
parser.add_argument("--rob-split", choices=["val", "test"], default="val",
                    help="--shift-test: split to perturb and score. 'test' is the frozen held-out split: use ONCE with the frozen model list, write the list down first, report it as post-freeze")
parser.add_argument("--rob-tag", default="", help="--shift-test: suffix for the output file name, e.g. _run2")
parser.add_argument("--curve", action="store_true", help="instead of the ratios, score ~45 log-spaced component counts from k=1 to the maximum -> pca_curve.csv")
parser.add_argument("--eval-test", action="store_true", help="also score the 21 exp test TMs (off by default: val only)")
parser.add_argument("--device", default="cpu", help="cpu keeps it off the GPU while training runs; cuda is much faster")
args = parser.parse_args()

LATENT_FOR_RATIO = {16: 32, 31: 16, 62: 8, 83: 6, 99: 5, 124: 4, 248: 2, 496: 1}
PIXELS, N_IN = 42025, 145
REAL_DIM = 2 * PIXELS          # 84,050
LATENT_AREA = 13 * 13
EVAL_TMS_PER_CHUNK = 4
device = torch.device(args.device)

EXP_SPLIT_DIR = paths.EXP_SPLIT_DIR
EXP_DATASET = paths.EXP_DATA
OUT_DIR = paths.OUT_ROOT / "pca_baseline"
RESULTS_CSV = OUT_DIR / "pca_results.csv"

exp_stack = np.load(EXP_DATASET, allow_pickle=True, mmap_mode="r")["TM"]


def load_tms(idx_file):
    return torch.tensor(exp_stack[np.load(EXP_SPLIT_DIR / idx_file)], dtype=torch.cfloat)  # [N, 42025, 145]


def to_patterns(tm):
    """[N, 42025, 145] complex -> [N*145, 84050] real (one row per input speckle pattern)."""
    p = tm.permute(0, 2, 1).reshape(-1, PIXELS)
    return torch.cat([p.real, p.imag], dim=1).float()


def to_tm(patterns, n_tm):
    c = torch.complex(patterns[:, :PIXELS], patterns[:, PIXELS:])
    return c.reshape(n_tm, N_IN, PIXELS).permute(0, 2, 1)


def sim_patterns(n_tm, seed=0, split="train"):
    """Speckle patterns of n_tm simulated composite-5m TMs (same forward model and noise as the autoencoder pretraining)."""
    sim_dir = paths.SIM_DIR
    idx_file = "train_composite5m_idx_2000.npy" if split == "train" else "val_composite5m_idx.npy"   # val: held-out simulated TMs
    sim_idx = np.load(paths.SIM_SPLIT_DIR / idx_file)[:n_tm]
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
    pool = torch.stack(pool).to(device)           # [P, 386, 386] 1m mode-basis TMs
    with open(sim_dir / "dist_pim_to_real.pkl", "rb") as f:
        dist_pim_to_real = torch.tensor(pickle.load(f)).to(device=device, dtype=torch.complex64)
    with open(sim_dir / "prox_real_to_pim.pkl", "rb") as f:
        prox_real_to_pim = torch.from_numpy(pickle.load(f)).to(device=device, dtype=torch.complex64)
    u_in = torch.tensor(np.load(paths.INPUTS_FILE),
                        dtype=torch.complex64, device=device)
    a_in = prox_real_to_pim @ u_in
    a_in_i = a_in / (torch.linalg.vector_norm(a_in, dim=0, keepdim=True) + 1e-8)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for s in range(0, n_tm, 4):
        r = rows[s:s + 4]
        T = pool[r[:, 0]]
        for j in range(1, 5):
            T = pool[r[:, j]] @ T
        X = dist_pim_to_real @ (T @ a_in_i)
        B = X.shape[0]
        amp_noise = torch.empty(B, 1, 1, device=device).uniform_(0.01, 0.15, generator=gen)
        phase_noise = torch.empty(B, 1, 1, device=device).uniform_(0.01, 0.1, generator=gen)
        amp, phase = X.abs(), X.angle()
        amp = amp + amp_noise * amp * torch.randn(amp.shape, device=device, generator=gen)
        phase = phase + phase_noise * torch.randn(phase.shape, device=device, generator=gen)
        X = amp * torch.exp(1j * phase)
        X = X / (torch.linalg.svdvals(X)[:, 0][:, None, None] + 1e-8)
        out.append(to_patterns(X.cpu()))
        print(f"  simulated {min(s + 4, n_tm)}/{n_tm} TMs", end="\r")
    print()
    return torch.cat(out)


def fit_pca(X, k_max):
    """X: [n_patterns, 84050] real training patterns (centred in place)."""
    X = X.to(device)
    n = X.shape[0]
    k_max = min(k_max, n - 1)
    mean = X.mean(0, keepdim=True)
    X -= mean
    t0 = time.time()
    G = X @ X.T                                   # [n, n]: eigendecomposition on the smaller side
    print(f"  Gram matrix {tuple(G.shape)} in {time.time() - t0:.0f}s")
    t0 = time.time()
    w, U = torch.linalg.eigh(G.double())          # torch only: scipy's eigh loads a second OpenMP runtime and crashes here
    w, U = w.flip(0)[:k_max], U.flip(1)[:, :k_max]
    del G
    print(f"  eigendecomposition in {time.time() - t0:.0f}s")
    keep = w > 1e-6 * w[0]                        # Gram entries are float32: eigenvalues below ~1e-6 of the largest are noise
    w, U = w[keep], U[:, keep]
    V = X.T @ U.float()
    V /= V.norm(dim=0, keepdim=True)              # [84050, k] unit-norm components (normalise by the measured norm, not sqrt(eigenvalue))
    print(f"  kept {V.shape[1]} components; max |V^T V - I| = {(V.T @ V - torch.eye(V.shape[1], device=V.device)).abs().max().item():.1e}")
    return mean, V


@torch.no_grad()
def score(mean, V, tm_eval, ks):
    sums = {k: [0.0, 0.0] for k in ks}
    for s in range(0, len(tm_eval), EVAL_TMS_PER_CHUNK):
        tm = tm_eval[s:s + EVAL_TMS_PER_CHUNK]
        xc = to_patterns(tm).to(device) - mean
        C = xc @ V
        for k in ks:
            recon = to_tm(C[:, :k] @ V[:, :k].T + mean, len(tm)).cpu()
            err = phase_aligned_loss_wholeTM_perSample(recon, tm)[1].item()
            r, t = recon.reshape(len(tm), -1).to(torch.complex128), tm.reshape(len(tm), -1).to(torch.complex128)
            corr = ((r * t.conj()).sum(1).abs() / (r.norm(dim=1) * t.norm(dim=1))).mean().item()
            sums[k][0] += err * len(tm)
            sums[k][1] += corr * len(tm)
    return {k: (e / len(tm_eval), c / len(tm_eval)) for k, (e, c) in sums.items()}


eval_sets = {"val": load_tms("val_split_idx.npy")}
if args.eval_test:
    eval_sets["test"] = load_tms("test_split_idx.npy")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_sources():
    """Yield (label, training patterns on CPU, a few of the fit TMs for the sanity score or None). One at a time to keep memory down."""
    for n in args.fit_n:
        if n == 0:
            continue
        idx_file = EXP_SPLIT_DIR / f"train_split_idx_{n}.npy"
        if n == 163:
            idx_file = EXP_SPLIT_DIR / "train_split_idx.npy"
        if idx_file.exists():
            idx = np.load(idx_file)
        else:                                          # sizes without a saved split (1, 2, 3): the splits are nested prefixes of the full list
            idx = np.load(EXP_SPLIT_DIR / "train_split_idx.npy")[:n]
        tm = torch.tensor(exp_stack[idx], dtype=torch.cfloat)
        yield f"exp{n}", to_patterns(tm), tm[:4]
    if args.fit_sim:
        yield f"sim{args.fit_sim}", sim_patterns(args.fit_sim), None


def sim_val_eval(label):
    """Held-out simulated composite-5m TMs (split 'simval'): in-domain for a sim-fitted PCA, the reverse domain gap for an exp-fitted one (--exp-on-sim)."""
    if args.sim_val and (label.startswith("sim") or args.exp_on_sim):
        return {"simval": to_tm(sim_patterns(args.sim_val, seed=1, split="val"), args.sim_val)}
    return {}


if args.shift_test:
    import json

    import torch.nn.functional as F
    from autoencoder_functions import TM_Autoencoder

    ae_dir = Path(args.ae_dir) if args.ae_dir else (paths.MODEL_ROOT / "final_62x" / "trial16" / "seed42")
    hp = json.loads((ae_dir / "hyperparameters.json").read_text())["params"]
    models = {}
    ae_files = [("AE sim-pretrained (ES weights)", "ES_best_weights.pt")]
    for n in args.ae_ft_n:                                                   # final runs save FT_best_weights_N{n}.pt; the older runs saved FT_best_weights.pt
        f = f"FT_best_weights_N{n}.pt" if (ae_dir / f"FT_best_weights_N{n}.pt").exists() else "FT_best_weights.pt"
        ae_files.append((f"AE + fine-tuned on {n} exp TMs", f))
    for item in args.rob_ae:
        label, file = item.rsplit("=", 1)
        ae_files.append((label, file))
    for name, file in ae_files:
        m = TM_Autoencoder(enc_channels=[hp["enc1"], 2 * hp["enc1"], 3 * hp["enc1"]], latent_dim=8,
                           attn_heads=hp["attn_heads"], linformer_k=hp["linformer_k"]).to(device)
        m.load_state_dict(torch.load(ae_dir / file, map_location=device))
        models[name] = m.eval()
    print(f"Loaded autoencoders from {ae_dir}")

    # Fixed in advance; every value is reported. Rotate/scale use bilinear resampling about the grid centre (zeros outside),
    # which also low-passes the speckle slightly - the blur row is the control for that.
    PERTURB_VALUES = {"shift": args.shifts, "rotate": [0, 1, 2, 5, 10], "scale": [1.0, 1.01, 1.02, 1.05, 1.1],
                      "blur": [0, 0.5, 1, 2], "noise": [0, 0.05, 0.1, 0.2]}

    def perturb(tm, kind, v):
        """tm: [N, 42025, 145] complex (CPU) -> perturbed copy, same shape."""
        if (kind in ("rotate", "blur", "noise") and v == 0) or (kind == "scale" and v == 1.0) or (kind == "shift" and v == 0):
            return tm
        out = []
        for i in range(0, len(tm), EVAL_TMS_PER_CHUNK):
            t = tm[i:i + EVAL_TMS_PER_CHUNK]
            B = len(t)
            if kind == "shift":
                s = int(v)
                t4 = t.reshape(B, 205, 205, N_IN)
                if args.shift_mode == "roll":
                    out.append(t4.roll(s, dims=2).reshape(B, PIXELS, N_IN))
                    continue
                shifted = torch.zeros_like(t4)                                       # crop: pixels past the edge are lost
                shifted[:, :, s:] = t4[:, :, :205 - s]
                if args.fill_level > 0 and s > 0:                                    # vacated strip: random noise, same for every model
                    rms = t4.abs().pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()
                    g = torch.Generator().manual_seed(i)
                    shape = shifted[:, :, :s].shape
                    noise = torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g)) / 2 ** 0.5
                    shifted[:, :, :s] = args.fill_level * rms * noise
                out.append(shifted.reshape(B, PIXELS, N_IN))
                continue
            if kind == "noise":
                rms = t.abs().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
                gen = torch.Generator().manual_seed(i)                                    # same noise for every model scored
                noise = torch.complex(torch.randn(t.shape, generator=gen), torch.randn(t.shape, generator=gen)) / 2 ** 0.5
                out.append(t + v * rms * noise)
                continue
            img = t.reshape(B, 205, 205, N_IN).permute(0, 3, 1, 2)                       # [B, 145, 205, 205] complex
            x = torch.cat([img.real, img.imag], dim=1).to(device)                         # [B, 290, 205, 205]
            if kind == "blur":
                r = int(np.ceil(3 * v))
                g = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=device, dtype=torch.float32) / v) ** 2)
                g = (g / g.sum())
                C = x.shape[1]
                x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
                x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
            else:
                ang = np.deg2rad(v) if kind == "rotate" else 0.0
                s = v if kind == "scale" else 1.0                                         # zoom in by s (sampling grid shrinks by 1/s)
                theta = torch.tensor([[np.cos(ang) / s, -np.sin(ang) / s, 0.0], [np.sin(ang) / s, np.cos(ang) / s, 0.0]],
                                     dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1, 1)
                grid = F.affine_grid(theta, x.shape, align_corners=False)
                x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            c = torch.complex(x[:, :N_IN], x[:, N_IN:]).cpu()
            out.append(c.permute(0, 2, 3, 1).reshape(B, PIXELS, N_IN))
        res = torch.cat(out)
        # keep each TM's overall scale as in the original, so a model isn't penalised just for a change in input amplitude
        return res * (tm.flatten(1).norm(dim=1) / res.flatten(1).norm(dim=1)).view(-1, 1, 1)

    @torch.no_grad()
    def ae_error(model, tm_eval):
        total = 0.0
        for i in range(0, len(tm_eval), EVAL_TMS_PER_CHUNK):
            X = tm_eval[i:i + EVAL_TMS_PER_CHUNK].to(device)
            B = len(X)
            X_in = torch.stack([X.real, X.imag], dim=1).reshape(B, 2, 205, 205, N_IN).permute(0, 1, 4, 2, 3)
            pred, _ = model(F.pad(X_in, (0, 3, 0, 3)), show_shapes=False)
            pred = pred[:, :, :, :205, :205].permute(0, 1, 3, 4, 2).reshape(B, 2, PIXELS, N_IN)
            pred_c = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())
            total += phase_aligned_loss_wholeTM_perSample(pred_c, X)[1].item() * B
        return total / len(tm_eval)

    if args.rob_split == "test":
        tm_val = load_tms("test_split_idx.npy")
    else:
        tm_val = eval_sets["val"]
    if args.rob_split == "test":
        print("*** ROBUSTNESS ON THE FROZEN TEST SPLIT: run once, with the model list written down beforehand ***")
    print(f"robustness on {len(tm_val)} {args.rob_split} TMs")

    results = []   # (perturbation, value, model, error)
    for kind in args.perturb:
        for name, m in models.items():
            for v in PERTURB_VALUES[kind]:
                results.append((kind, v, name, ae_error(m, perturb(tm_val, kind, v))))
                print(f"  {name} | {kind} {v}: {100 * results[-1][3]:.2f}%")
    ks_requested = {r: LATENT_FOR_RATIO[r] * LATENT_AREA for r in args.ratios}
    for label, X_fit, _ in fit_sources():
        mean, V = fit_pca(X_fit, max(ks_requested.values()))
        del X_fit
        for kind in args.perturb:
            for v in PERTURB_VALUES[kind]:
                res = score(mean, V, perturb(tm_val, kind, v), sorted({min(k, V.shape[1]) for k in ks_requested.values()}))
                for r, k in ks_requested.items():
                    results.append((kind, v, f"PCA {label} {r}x", res[min(k, V.shape[1])][0]))
                    print(f"  PCA {label} {r}x | {kind} {v}: {100 * results[-1][3]:.2f}%")
        del mean, V
    # earlier runs used the circular roll and wrote pca_robustness_test.csv; crop mode writes its own file so nothing is overwritten
    shift_csv = OUT_DIR / (("pca_robustness_test" if args.shift_mode == "roll" else "pca_robustness_test_crop") + args.rob_tag
                           + ("_TESTSPLIT" if args.rob_split == "test" else "") + ".csv")   # the old file names say 'test' but hold val results
    with open(shift_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["perturbation", "value", "model", f"{args.rob_split}_error"])
        writer.writerows([(k, v, m, round(e, 5)) for k, v, m, e in results])
    print(f"Saved to {shift_csv}")
    sys.exit(0)

if args.curve:
    # Error vs number of components. Components past the float32 noise floor are dropped by fit_pca, so the curve ends where PCA does.
    CURVE_CSV = OUT_DIR / "pca_curve.csv"
    new_curve = not CURVE_CSV.exists()
    with open(CURVE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_curve:
            writer.writerow(["fit_n", "split", "k", "ratio", "error", "complex_corr", "n_eval_tms"])
        for label, X_fit, tm_sanity in fit_sources():
            print(f"Fitting PCA on {label} ({X_fit.shape[0]} speckle patterns)")
            mean, V = fit_pca(X_fit, 6000)
            del X_fit
            ks = sorted(set(np.geomspace(1, V.shape[1], 45).round().astype(int).tolist())
                        | {k for k in (LATENT_FOR_RATIO[r] * LATENT_AREA for r in LATENT_FOR_RATIO) if k <= V.shape[1]})
            for split, tm_eval in {**eval_sets, **({"fit": tm_sanity} if tm_sanity is not None else {}), **sim_val_eval(label)}.items():
                res = score(mean, V, tm_eval, ks)
                for k in ks:
                    err, corr = res[k]
                    writer.writerow([label, split, k, round(REAL_DIM / k, 2), round(err, 5), round(corr, 5), len(tm_eval)])
                f.flush()
                print(f"  fit {label} | {split}: " + ", ".join(f"k={k}: {100 * res[k][0]:.1f}%" for k in ks[::6]))
            del mean, V
    print(f"Saved to {CURVE_CSV}")
    sys.exit(0)

new_file = not RESULTS_CSV.exists()
with open(RESULTS_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["fit_n", "split", "ratio", "k_requested", "k_used", "error", "complex_corr", "n_eval_tms"])
    for label, X_fit, tm_sanity in fit_sources():
        print(f"Fitting PCA on {label} ({X_fit.shape[0]} speckle patterns)")
        ks_requested = {r: LATENT_FOR_RATIO[r] * LATENT_AREA for r in args.ratios}
        mean, V = fit_pca(X_fit, max(ks_requested.values()))
        del X_fit
        ks_used = {r: min(k, V.shape[1]) for r, k in ks_requested.items()}
        # "fit" = 4 of the exp TMs PCA was fitted on: a sanity check (should be far lower than val, ~0 when k is close to the sample count)
        for split, tm_eval in {**eval_sets, **({"fit": tm_sanity} if tm_sanity is not None else {}), **sim_val_eval(label)}.items():
            res = score(mean, V, tm_eval, sorted(set(ks_used.values())))
            for r in args.ratios:
                err, corr = res[ks_used[r]]
                note = "" if ks_used[r] == ks_requested[r] else f" (only {ks_used[r]} components available)"
                print(f"  fit {label} | {split} | {r:>3}x (k={ks_requested[r]}): error {100 * err:6.2f}%, corr {corr:.4f}{note}")
                writer.writerow([label, split, r, ks_requested[r], ks_used[r], round(err, 5), round(corr, 5), len(tm_eval)])
                f.flush()
        del mean, V
print(f"Saved to {RESULTS_CSV}")
