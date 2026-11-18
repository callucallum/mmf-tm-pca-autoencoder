# Simulation-pretrained autoencoder vs PCA for multimode-fibre transmission matrices

Code, trained weights and split indices for

> **Benchmarking a simulation-pretrained autoencoder against PCA for compressing multimode fibre transmission matrices from few measurements**
> C. J. Wilson et al. (arXiv: *TODO – add link when the preprint is online*)

A convolutional autoencoder with channel attention compresses a 50 µm multimode-fibre transmission matrix (TM) 62-fold. It is pretrained on
simulated fibres (composites of five simulated 1 m TMs), then fine-tuned on a few measured TMs. It is benchmarked against per-speckle PCA
fitted on the same measured TMs. The paper finds the autoencoder is **not** a better compressor than PCA once about ten real TMs are available. Its value is
as a prior when few or no real TMs exist. This repository contains everything needed to reproduce the original-fibre experiments.

> **Not reproducible from this repository:** the paper also tests transfer to a larger-core fibre measured by another group (Wen et al.).
> Those TMs are not ours to release, so none of the collaborator-fibre code or weights is included. The collaborator rows of the paper's tables
> and figures cannot be reproduced here.

## Layout

| folder | contents |
|---|---|
| `src/` | `autoencoder_functions.py` (model, phase-aligned loss), `paths.py` (all data/output locations, see below) |
| `splits/` | scripts that build the frozen splits, and the resulting index files in `splits/indices/{exp,sim}/` |
| `training/` | baseline-recipe scripts (`PT_`, `FT_`, `ES_`, `ES_FT_` with their `run_*` drivers) and the final-configuration scripts (`Optuna_62x_ES_FT.py`, `Final_62x_ES_FT.py`) |
| `evaluation/` | `pca_baseline.py`, `fewshot_finetune.py`, `final_test_eval.py` (frozen test split, run once), `unscramble_test.py` (focusing and imaging through reconstructed TMs) |
| `optuna/` | the Optuna study (27 trials) from which trial 16 was chosen |
| `weights/` | trained weights of the final configuration (see below) |

## Data

**Experimental TMs.** 204 TMs of a 5 m, 50 µm, NA 0.22 step-index fibre with random bends (660 nm), 145 inputs each, are released as raw
`TM.mat` files (together with the corresponding reflection matrices) at [huggingface.co/datasets/OPTIMlab/mmf-tm-rm-experimental](https://huggingface.co/datasets/OPTIMlab/mmf-tm-rm-experimental)
(CC BY 4.0, DOI [10.57967/hf/11679](https://doi.org/10.57967/hf/11679)).

```bash
git clone https://huggingface.co/datasets/OPTIMlab/mmf-tm-rm-experimental
cd mmf-tm-rm-experimental
python Import_and_Prepare_TMs.py   # writes processed_dataset.npz (keys 'TM' (204, 42025, 145) and 'RM') and processed_dataset_paths.txt
```

`processed_dataset.npz` holds the pixel-basis TMs (205 x 205 output pixels, 145 inputs), each divided by its largest singular value; the scripts here use only `TM`. The processing script needs about 52 GB of RAM
and the `.npz` is about 25 GB. The row order is the order of
`processed_dataset_paths.txt`, and the frozen split indices in `splits/indices/exp/` refer to that order. By default the scripts look for
`../mmf-tm-rm-experimental/processed_dataset.npz` (i.e. the two repositories side by side); otherwise set `TM_EXP_DATA`.

**Simulated TMs.** The pretraining pool of 5,000 simulated 1 m TMs (`TM*.pkl`), `dist_pim_to_real.pkl`, `prox_real_to_pim.pkl` and the 145 orthonormal input patterns
(`orthonormal_random_inputs_371x371_145.npy`) are released as pre-generated files in the `TransferMatrixMethod/` folder of [huggingface.co/datasets/OPTIMlab/mmf-tm-simulated](https://huggingface.co/datasets/OPTIMlab/mmf-tm-simulated)
(CC BY 4.0, DOI [10.57967/hf/11677](https://doi.org/10.57967/hf/11677)). The simulator itself is not part of this repository.

```bash
git clone https://huggingface.co/datasets/OPTIMlab/mmf-tm-simulated
```

By default the scripts read them from `../mmf-tm-simulated/TransferMatrixMethod/`; otherwise set `TM_SIM_DIR` (and `TM_INPUTS` if the input-pattern file is elsewhere). Then check the splits:

```bash
python splits/train_test_split_sim_indices_5m.py                        # checks the released composite-5 m indices against a fresh regeneration
python splits/train_test_split_exp_indices.py                           # same for the experimental split (confirms the procedure and TM count; the row order is fixed by processed_dataset_paths.txt)
```

The released index files in `splits/indices/` fix which pool entries form each train/validation/test composite, and `weights/` lets you check the downstream numbers against our exact models.

All locations can be changed with environment variables (full list in `src/paths.py`): `TM_EXP_DATA`, `TM_SIM_DIR` (default `../mmf-tm-simulated/TransferMatrixMethod`), `TM_INPUTS`,
`TM_SIM_SPLIT_DIR`, `TM_EXP_SPLIT_DIR`, `TM_OUT_ROOT` (training outputs, default `outputs/`), `TM_MODEL_ROOT` (where evaluation scripts read weights, default `weights/`)
and `TM_OPTUNA_DB`.

## Order of steps

1. **Download the data and check the splits** (above).
2. **Baseline recipe** (124x and 62x, `training/PT_*`, `ES_*`, `FT_*`, `ES_FT_*`; used in Sections 3.1-3.2 of the paper). Each `*_Autoencoder.py` trains one sub-batch of 100 composite TMs,
   and its `run_*` driver loops over sub-batches and epochs and applies early stopping. Seed, simulated set size and the number of experimental TMs are set at the top of each script and driver
   (they must match). Checkpoints go to `outputs/5m_fibre/{124x,62x}_compression/seed*/<n>/`.
3. **Final configuration** (62x, 5 seeds; Sections 3.3 onwards):
   ```bash
   python training/Optuna_62x_ES_FT.py                       # hyperparameter search, writes/resumes TM_OPTUNA_DB (trial 16 was chosen)
   python training/Final_62x_ES_FT.py --trials 16 --seeds 42 43 44 45 46 --exp-ft-n 5 10 163 \
          --max-pretrain-epochs 5000 --max-ft-epochs 5000 --out-name final_62x
   python training/Final_62x_ES_FT.py --trials 16 --seeds 42 43 44 45 46 --exp-ft-n 163 10 5 \
          --exp-only --exp-batch-size 2 --max-pretrain-epochs 5000 --out-name exp_only_62x     # from-scratch baseline, no simulation
   ```
   These command lines are reconstructed from the `hyperparameters.json` files saved with each run; `Final_62x_ES_FT.py --help` lists every option.
   Output goes to `outputs/final_62x/trial16/seed*/`. To evaluate your own checkpoints instead of the released ones, set `TM_MODEL_ROOT` to the
   same folder as `TM_OUT_ROOT` (e.g. `TM_MODEL_ROOT=outputs`).
4. **PCA baseline** and robustness: `python evaluation/pca_baseline.py --fit-n 5 10 163 --device cuda` (add `--curve` for the error-vs-components curve, `--fit-sim 163` for PCA fitted on simulation, `--shift-test --perturb shift rotate scale blur noise` for the drift tests).
5. **Few-shot specialist** (2 real TMs): `python evaluation/fewshot_finetune.py --ks 2 --seeds 42 --save-weights`.
6. **Frozen test evaluation** (21 original-fibre test TMs, scored once): `python evaluation/final_test_eval.py --plan`, then `--device cuda`. A lock file prevents a second run.
7. **Focusing and imaging**: `python evaluation/unscramble_test.py --pattern F --split test` (also `ring`, `spots`).

## Weights

`weights/final_62x/trial16/seed42 ... seed46/` (final configuration, Optuna trial 16: widths 24/48/72, 1 head, Linformer k = 24, 62x compression):

| file | model |
|---|---|
| `ES_best_weights.pt` | pretrained on 2,000 simulated composite 5 m TMs (checkpoint chosen on the 20 experimental validation TMs) |
| `FT_best_weights_N{5,10,163}.pt` | the above fine-tuned on the first 5 / 10 / all 163 experimental training TMs |
| `FewShot_exp_N163.pt` | fine-tuned on all 163 training TMs with the few-shot recipe (seed 42 only has `FewShot_exp_N2.pt`: fine-tuned on 2 TMs, used in the paper's test tables) |
| `hyperparameters.json` | architecture and training hyperparameters |

`weights/exp_only_62x/trial16/seed42 ... seed46/`: the same architecture trained from scratch on 5 / 10 / 163 experimental TMs (`ExpOnly_best_weights_N*.pt`), no simulation.
Total size about 58 MB. The paper's test results use seed 42 only. Weights from the earlier baseline-recipe experiments (124x, 500/1,000/2,000 composites) are not included.

```python
import json, torch
from src.autoencoder_functions import TM_Autoencoder

d = "weights/final_62x/trial16/seed42"
hp = json.load(open(f"{d}/hyperparameters.json"))["params"]
model = TM_Autoencoder(enc_channels=[hp["enc1"], 2 * hp["enc1"], 3 * hp["enc1"]], latent_dim=8,   # latent_dim 8 = 62x
                       attn_heads=hp["attn_heads"], linformer_k=hp["linformer_k"])
model.load_state_dict(torch.load(f"{d}/ES_best_weights.pt", map_location="cpu"))
model.eval()
# input: [B, 2, 145, 208, 208] (real/imag, input pattern, output image zero-padded from 205 to 208); returns (reconstruction, latent)
```

## Environment used

Windows 11, NVIDIA RTX Pro 6000 Blackwell Max-Q Workstation Edition, Python 3.12.14, PyTorch 2.14.0+cu130. Training of the final configuration used deterministic cuDNN/cuBLAS kernels with TF32
disabled (`--fast` turns this off). See `requirements.txt`.

## Citation and licence

Citation: *TODO – add BibTeX when the preprint is online.*

Licence: code under the [MIT licence](LICENSE); trained weights in `weights/` under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), like the two datasets.
