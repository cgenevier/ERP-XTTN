# ERP-XTTN: Interpretable Cross-Attention ERP Classifier

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20087550-blue)](https://doi.org/10.5281/zenodo.20087550)
[![arXiv](https://img.shields.io/badge/arXiv-2606.02939-b31b1b.svg)](https://arxiv.org/abs/2606.02939)

Code repository for ERP-XTTN, an interpretable cross-attention architecture for ERP classification from EEG signals.

> **Version 2.0.0** — This release extends the v1.0.0 *Graz BCI 2026* work to the full ERP CORE benchmark (seven paradigms) alongside the BNCI and HRI ErrP datasets, adding EEGNet and xDAWN+RG baselines and an auto peak-detection ERPXTTN variant. It accompanies the ERP-XTTN extension paper (preprint: [arXiv:2606.02939](https://arxiv.org/abs/2606.02939)). For the frozen Graz conference release, see the [`v1.0.0` release](https://github.com/cgenevier/ERP-XTTN/releases/tag/v1.0.0).

## Versions

- **[v2.0.0](https://github.com/cgenevier/ERP-XTTN/releases/tag/v2.0.0)** *(this release)* — Extension paper covering the full ERP CORE
  benchmark (seven paradigms) plus the BNCI and HRI ErrP datasets, with EEGNet
  and xDAWN+RG baselines and constrained/auto ERPXTTN variants.
  - Preprint: [arXiv:2606.02939](https://arxiv.org/abs/2606.02939)
  - Archived on Zenodo: [10.5281/zenodo.20497891](https://doi.org/10.5281/zenodo.20497891)
- **[v1.0.0](https://github.com/cgenevier/ERP-XTTN/releases/tag/v1.0.0)** — Code for Wyman & Hirshfield, *Graz BCI 2026* (conference paper).
  - Paper DOI: [pending]
  - Archived on Zenodo: [10.5281/zenodo.20087551](https://doi.org/10.5281/zenodo.20087551)

To reproduce the Graz conference results, check out the [`v1.0.0`](https://github.com/cgenevier/ERP-XTTN/releases/tag/v1.0.0) tag:
`git checkout v1.0.0`

## Repository Structure

```
ERP-XTTN/
├── 01_convert_data.py      # Raw data → MNE Raw FIF + BIDS export
├── 02_inspect.py           # Data inspection / quality check utilities
├── 03_preprocess.py        # Preprocessing + epoching pipeline
├── 04_train.py             # LOSO cross-validation training
├── 05_gen_figures.py       # Per-dataset attention & TP/TN figure generation
├── 06_gen_analysis.py      # Cross-dataset interpretability analysis → analysis_summary.json
├── 07_gen_paper_figures.py # Aggregate multi-dataset figures for the paper
├── bench_latency.py        # Inference latency / parameter-count benchmark
├── run_manager.py          # Batch launcher / status monitor for experiment runs
├── dashboard.html          # Browser dashboard for browsing saved results
├── eegnet.py               # EEGNet baseline (Lawhern et al., 2018)
├── eeg_deformer.py         # EEG-Deformer baseline (Ding et al., 2025)
├── epmn.py                 # ERP Prototypical Matching Net baseline (Wei et al., 2022)
├── erpxttn.py              # ERP-XTTN model + automatic peak detection
├── xdawn_rg.py             # xDAWN + Riemannian Geometry baseline
├── analysis_summary.json   # Cached cross-dataset analysis output (from 06)
├── paper_figures/          # Paper figures + architecture diagram source
│   ├── extension/          #   v2.0.0 extension-paper figures + make_diagram.py
│   └── graz2026/           #   v1.0.0 Graz conference figures
├── datasets/
│   ├── bnci_errp_013-2015/         # BNCI Horizon 2020 ErrP (feedback)
│   ├── hri_errp_cursor/            # HRI cursor ErrP (feedback)
│   ├── erpcore_ern/                # ERP CORE — ERN (response-locked)
│   ├── erpcore_lrp/                # ERP CORE — LRP
│   ├── erpcore_mmn/                # ERP CORE — MMN
│   ├── erpcore_n170/               # ERP CORE — N170
│   ├── erpcore_n2pc/               # ERP CORE — N2pc
│   ├── erpcore_n400/               # ERP CORE — N400
│   └── erpcore_p300/               # ERP CORE — P300
└── logs/                    # Generated at run time (not tracked in git)
    └── <timestamp>_<dataset>_<channels>_<model>/
        └── train.log
```

Each dataset directory contains:
- `dataset_config.json` — format, event IDs, channel presets, label mapping, polarity pattern
- `original_data/` — raw source files (not tracked in git)
- `raw_fif/` — converted MNE Raw FIF files
- `qc_outputs/` — quality check outputs from inspect
- `epoched_fif/` — preprocessed epochs
- `results/` — model outputs (predictions, attention weights, figures)

## Models

| Model | Description |
|-------|-------------|
| **EEGNet** | Lawhern et al. (2018) compact CNN baseline |
| **EEG-Deformer** | Ding et al. (2025) dense convolutional transformer baseline (faithful port of the official implementation) |
| **EPMN** | Wei et al. (2022) ERP Prototypical Matching Net — metric-based meta-learning baseline (episodic training) |
| **xDAWN+RG** | xDAWN spatial filtering + Riemannian geometry classifier (classical ML baseline) |
| **ERPXTTN Constrained** | Cross-attention prototype model with dataset-configured polarity pattern; results saved under `erpxttn_constrained/` |
| **ERPXTTN Auto** | Auto peak-detection variant of ERPXTTN; results saved under `erpxttn_auto/` |

All neural models are trained under the shared LOSO two-phase protocol (Section 2.5 of the paper); EPMN additionally uses its native episodic meta-learning. Results for stochastic models are produced across **5 seeds**; the deterministic xDAWN+RG uses one. Per-seed outputs are written under `.../<model>/seed-<N>/` and aggregated by `06_gen_analysis.py` and `07_gen_paper_figures.py`.

## Requirements

- Python 3.10+ (tested with Python 3.11.14)
- CUDA-capable GPU recommended (experiments used an NVIDIA RTX PRO 6000 Blackwell Server Edition via RunPod; v1.0.0 results were on an NVIDIA RTX 4070 Laptop GPU)

Install dependencies:

```bash
pip install -r requirements.txt
```

For data conversion only (step 2), additional packages:
```bash
pip install mne-bids pybv
```

## Steps to Reproduce

### 1. Data Acquisition

**BNCI Horizon 2020 — Monitoring Error-Related Potentials** (Margaux et al., 2012)
- 6 subjects, 64 EEG channels, 512 Hz
- Publicly available at https://bnci-horizon-2020.eu/database/data-sets
- Place raw `.mat` files in `datasets/bnci_errp_013-2015/original_data/`

**HRI Cursor** (Ehrlich & Cheng, 2019)
- 11 subjects (sub-02 through sub-13, excluding sub-01 and sub-12), 27 EEG channels, 256 Hz
- Publicly available at https://github.com/stefan-ehrlich/dataset-ErrP-HRI/
- Place raw `.set/.fdt` files in `datasets/hri_errp_cursor/original_data/`

**ERP CORE** (Kappenman et al., 2021)
- 40 subjects, 30 EEG channels, 1024 Hz
- Seven paradigms: ERN, LRP, MMN, N170, N2pc, N400, P300
- Publicly available at https://osf.io/thsqg/
- Place per-paradigm raw files in `datasets/erpcore_<paradigm>/original_data/`

Each dataset directory requires a `dataset_config.json` specifying format, event IDs, channel presets, label mapping, and polarity pattern. Training subject lists and result variant names are defined in `04_train.py` and `05_gen_figures.py`.

### 2. Convert to BIDS + FIF

```bash
python 01_convert_data.py datasets/bnci_errp_013-2015
python 01_convert_data.py datasets/hri_errp_cursor
python 01_convert_data.py datasets/erpcore_n400   # repeat per ERP CORE paradigm
```

### 3. Inspect (Optional QC)

```bash
python 02_inspect.py datasets/erpcore_n400
```

### 4. Preprocess and Epoch

Preprocessing applies: optional re-referencing, channel selection, IIR bandpass (4th-order Butterworth, forward-phase; cutoffs per dataset config), optional resampling, and epoching (window per dataset config). The `--no-qc-drop` flag reproduces the variant names used by the checked-in results and training scripts.

```bash
# ErrP datasets (resampled to 256 Hz)
python 03_preprocess.py --dataset datasets/bnci_errp_013-2015 --channels midline3 --resample 256 --no-qc-drop
python 03_preprocess.py --dataset datasets/hri_errp_cursor    --channels midline3 --resample 256 --no-qc-drop

# ERP CORE — paradigm-specific channel presets + full 30ch
python 03_preprocess.py --dataset datasets/erpcore_n400 --channels midline3_n400 --resample 256 --no-qc-drop
python 03_preprocess.py --dataset datasets/erpcore_n400 --channels full          --resample 256 --no-qc-drop
# ...repeat per paradigm
```

### 5. Train (LOSO Cross-Validation)

```bash
# Baselines (one seed shown; sweep uses --seed 1..5 for neural models)
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model eegnet        --seed 1
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model eeg_deformer  --seed 1
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model epmn          --seed 1
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model xdawn_rg      --seed 1   # deterministic

# ERPXTTN — auto peak-detection (default; the model reported in the paper)
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn       --seed 1

# ERPXTTN — constrained variant (dataset-configured polarity pattern; the v1.0.0 model)
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn --peak-mode constrained

# ERPXTTN ablations (write to a separate erpxttn_auto_<tag>/ dir via --ablation-tag)
python 04_train.py --dataset hri_errp_cursor --channels midline3 --model erpxttn --xattn-mode qkv --ablation-tag qkv   --seed 1
python 04_train.py --dataset hri_errp_cursor --channels midline3 --model erpxttn --no-self-attn    --ablation-tag nosa --seed 1
python 04_train.py --dataset hri_errp_cursor --channels midline3 --model erpxttn --max-k 2         --ablation-tag k2   --seed 1
```

`--model erpxttn` selects the ERPXTTN architecture and **defaults to auto peak-detection** (saved under `erpxttn_auto/`; accepts `--max-k` to cap the number of detected prototypes, default 4). Pass `--peak-mode constrained` for the v1.0.0 constrained variant (saved under `erpxttn_constrained/`). Ablation knobs (`--xattn-mode qkv`, `--no-self-attn`, `--num-heads`, `--patch-width`, `--d-model`, `--peak-prominence`) require an `--ablation-tag` so runs land in their own subdirectory.

Results are saved to `datasets/<name>/results/<window>/<variant>/<model>/seed-<N>/results.json`. The full multi-seed sweep (all models × datasets × montages × 5 seeds, plus the ablation grid) is orchestrated by `run_manager.py` — see the reproduction section below.

### 6. Generate Figures (ERPXTTN only)

```bash
python 05_gen_figures.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn_auto
```

Generates per-subject attention routing figures (TP/TN high-confidence and median trials) and aggregate attention analysis plots (prototype visualization, entropy vs. AUROC, attention timecourse, differential overlay, per-subject routing). Use `--morphology-only` to regenerate just the morphology panels, or `--partial` to skip datasets whose results are incomplete.

### 7. Cross-Dataset Analysis and Paper Figures

The aggregate (multi-dataset) analysis and publication figures are driven by the auto ERPXTTN runs (`erpxttn_auto/`):

```bash
# Cross-dataset interpretability analysis → analysis_summary.json
python 06_gen_analysis.py

# Aggregate paper figures (reads analysis_summary.json + per-subject results) → paper_figures/extension/
python 07_gen_paper_figures.py
```

`bench_latency.py` reports inference latency and parameter counts for the model/baselines.

### Reproducing the Paper Results

**All ERPXTTN results reported in the extension paper use the automatic peak-detection variant (`erpxttn_auto`)**, which is the default for `04_train.py` and `run_manager.py` in this release. The constrained variant (`erpxttn_constrained`) is the v1.0.0 model and is kept only for comparison; reach it with `--peak-mode constrained`. The paper uses the 3-channel preset per dataset (the cross-dataset analysis additionally uses `--channels full`):

| Dataset | `--channels` preset |
|---------|---------------------|
| bnci_errp_013-2015 | midline3 |
| hri_errp_cursor | midline3 |
| erpcore_ern | midline3_ern |
| erpcore_lrp | lateral3_lrp |
| erpcore_mmn | midline3 |
| erpcore_n170 | occipital3_n170 |
| erpcore_n2pc | posterior3_n2pc |
| erpcore_n400 | midline3_n400 |
| erpcore_p300 | midline3 |

The full sweep is 5 seeds × {EEGNet, EEG-Deformer, EPMN, ERP-XTTN} + 1-seed xDAWN+RG, across all datasets and both montages, plus the ERP-XTTN ablation grid (QK-only vs QKV, self-attention on/off, K, peak prominence, heads) on HRI/P300/N400 at 3 channels. `run_manager.py` enumerates and launches it:

```bash
# Batch-launch the whole matrix (3ch + full + ablations), respecting MAX_GPU_JOBS.
python run_manager.py --launch-all --include-full --include-ablation

# Or run as a daemon that keeps GPU slots filled:
python run_manager.py --daemon --include-full --include-ablation

# Then aggregate seed-averaged analysis + stats + paper figures:
python 06_gen_analysis.py      # -> analysis_summary.json + stats_summary.csv (Wilcoxon + effect sizes + BH-FDR)
python 07_gen_paper_figures.py # -> paper_figures/ (incl. per-subject AUROC point figures)
```

A single (dataset, preset, model, seed) can also be run directly, e.g.:

```bash
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn --seed 3
```

`06_gen_analysis.py` aggregates each subject's AUROC over all available seeds before computing the paired Wilcoxon signed-rank tests, rank-biserial effect sizes, and Benjamini–Hochberg FDR correction, and reports per-seed dataset-mean AUROC as a stability check.

## Architecture

### ERP-XTTN

ERP-XTTN classifies EEG trials by cross-attending input signal patches against ERP prototype templates derived from the grand-average difference wave. The architecture has five stages:

**1. Patch Embedding + Positional Encoding**

The input epoch (C channels × T time samples) is divided into N = T / w non-overlapping temporal patches of width w = 8 samples (31.25 ms at 256 Hz). Each patch is flattened to a C·w-dimensional vector and linearly projected to d = 64 dimensions. Learned positional embeddings are added to encode temporal position.

**2. Self-Attention**

A single multi-head self-attention layer (H = 4 heads, d_h = 16) with pre-layer-normalization and residual connection refines the patch embeddings, allowing patches to share temporal context before cross-attention.

**3. Prototype Construction (per fold, per training phase)**

For each LOSO fold, K ERP prototypes are extracted from the grand-average difference wave on the dataset's detection channel. Extraction runs once in Phase 1 (train split) and again in Phase 2 (full training pool). Two extraction modes are available:

- **Auto (default; the model reported in the paper):** the difference wave is Gaussian-smoothed (σ = 2 samples), and the top-K local extrema by prominence (above the dataset-configured threshold) are selected, with no polarity constraint. K is the number of peaks actually detected and may vary by fold (capped by `--max-k`, default 4).
- **Constrained (the v1.0.0 model, retained for comparison):** peaks are instead selected to match a dataset-configured expected polarity pattern while maximizing total prominence, fixing K per dataset (see [Prototype Configuration](#prototype-configuration)).

In both modes:

- Window boundaries expand from each peak to neighboring zero-crossings, clamped to [40, 200] ms width.
- Each prototype is the segment of the full multichannel difference wave within its detected window, zero-padded elsewhere to epoch length.

Prototypes are embedded through the same patch embedding layer as input patches and mean-pooled across the patch dimension to produce K vectors of dimension d. Shared positional encoding (indexed at each prototype's temporal center) is added, placing prototypes in the same positional space as input patches.

**4. QK-Only Cross-Attention**

Input patch embeddings serve as queries (Q) and prototype embeddings as keys (K). Separate layer norms are applied before Q and K linear projections. Scaled dot-product attention produces weights in (B, H, N, K) — no value (V) projection is used. The attention map directly encodes how each input patch routes to each prototype.

**5. Classification**

Attention weights are averaged across heads, flattened to N·K dimensions, and passed through a single linear layer to produce a scalar logit (or multi-way logits for paradigms with >2 classes).

### EEGNet (Baseline)

EEGNet (Lawhern et al., 2018) is used as a compact CNN baseline. It applies temporal convolution, depthwise spatial convolution, separable convolution, and average pooling, with max-norm weight constraints. Implementation follows Table 2 of the original paper with F1=8, D=2, F2=16, dropout=0.25.

### EEG-Deformer (Baseline)

EEG-Deformer (Ding et al., 2025) is a dense convolutional transformer: a CNN shallow encoder, a stack of Hierarchical Coarse-to-fine Transformer (HCT) blocks with a fine-grained temporal-learning branch, and dense information-purification (IP) units feeding an MLP head. `eeg_deformer.py` is a faithful port of the official implementation (einops replaced with plain torch, no new dependency); the temporal kernel is set to the nearest odd integer to 0.1·fs (27 at 256 Hz), other hyperparameters are the paper defaults (depth 4, 16 heads, 64 kernels). Trained under the shared protocol.

### EPMN (Baseline)

ERP Prototypical Matching Net (Wei et al., 2022) is a metric-based meta-learning baseline for zero-calibration ERP classification, and the closest published prototype-based competitor to ERP-XTTN. `epmn.py` reimplements the Manor-CNN feature extractor (paper Table 2), squared-Euclidean prototype distance, softmax attention kernel, and the classification + metric losses (Eqs. 1–8). Training (`run_fold_epmn` in `04_train.py`) is faithful episodic meta-learning: each episode uses one training subject as the query domain and builds class prototypes from the remaining (support) subjects' ERP templates. Preprocessing, channels, folds, seeds, the two-phase early-stopping protocol, and evaluation are held identical to the other models; the subject-level validation split is intrinsic to meta-learning.

### xDAWN + Riemannian Geometry (Baseline)

A classical ML baseline: xDAWN spatial filters estimated on the training set project epochs onto ERP-enhanced components, their trial covariance matrices are projected to the Riemannian tangent space, and a logistic-regression classifier (L2, C=1.0, L-BFGS) is fit. Implemented via `pyriemann` + scikit-learn.

### Training Procedure

- **Cross-validation**: Leave-one-subject-out (LOSO)
- **Two-phase training per fold**:
  - Phase 1: Train on 85% of pooled leave-out data with 15% stratified validation split. Early stopping (patience=15) on validation AUROC determines best epoch count.
  - Phase 2: Retrain from scratch on the full training pool for exactly best_epoch+1 epochs. Evaluate on the held-out test subject.
  - ERP-XTTN prototype extraction: Run in both phases, using the phase-specific training data.
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **LR schedule**: Linear warmup (5 epochs, from lr/10) + cosine annealing over 100 epochs (to lr/100)
- **Loss**: BCEWithLogitsLoss with inverse class-frequency pos_weight (binary) / CrossEntropy (multi-class)
- **Augmentation** (training only): Temporal jitter (uniform ±10 samples) + Gaussian noise (σ=0.1)
- **Normalization**: Per-channel z-score computed on training data, applied to train/val/test
- **Gradient clipping**: Max norm = 1.0
- **Batch size**: 128 (on the current GPU; v1.0.0 results used 32 — deltas validated within noise)
- **Seed**: 42 (deterministic)

## Prototype Configuration

The following parameters are configured per dataset in `dataset_config.json`, informed by established ERP morphology:

- **Peak prominence**: Minimum prominence threshold (in z-scored units) for peak detection. Tuned per dataset to ensure the detector anchors on canonical ERP components rather than small early deflections. Applies to **both** variants.
- **Polarity pattern**: The expected alternating sign pattern of ERP components (e.g., `neg, pos` for a negative-then-positive waveform). Used **only by the constrained variant**, where its length fixes the number of prototypes (K). The auto variant (the model reported in the paper) ignores the polarity pattern and instead takes the top-K most prominent peaks, so its K is detected per fold (capped by `--max-k`, default 4).
- **Proto names**: Human-readable component labels for each constrained-variant prototype window.

Prototype windows are detected automatically per LOSO fold from the training data only — no test-set information is used.

The table below lists the **constrained-variant** configuration. The **Auto K** column gives the auto variant's observed modal K per fold (from `analysis_summary.json`); because auto does not enforce the polarity pattern, it differs from the constrained K for several datasets, so its values should not be read off the *Pattern* / *Constrained K* columns.

| Dataset | Pattern (constrained) | Constrained K | Auto K (modal) | Prominence | Prototypes (constrained) |
|---------|---------|---|---|------------|------------|
| BNCI ErrP | pos, neg, pos, neg | 4 | 4 | 0.02 | P1-diff, Ne-diff, Pe-diff, LateN-diff |
| HRI ErrP | pos, neg, pos, neg | 4 | 4 | 0.02 | P1-diff, Ne-diff, Pe-diff, LateN-diff |
| ERN | neg, pos | 2 | 3 | 0.02 | ERN-diff, Pe-diff |
| LRP | neg, pos | 2 | 2 | 0.02 | EarlyN, LateP |
| MMN | neg, pos, neg | 3 | 3 | 0.02 | MMN-diff, P3a-diff, LateN-diff |
| N170 | neg, pos, neg | 3 | 4 | 0.02 | N170-diff, VPP-diff, LateN-diff |
| N2pc | neg, pos, neg | 3 | 4 | 0.02 | N2pc-diff, SPCN-diff, LateN-diff |
| N400 | pos, neg, pos, neg | 4 | 4 | 0.02 | P2-diff, N400-diff, LPC-diff, LateN-diff |
| P300 | pos, neg, pos | 3 | 4 | 0.02 | P3-diff, SW-diff, LateP-diff |

Auto K is the modal value across folds (consistent for all datasets except N400 and P300, which vary between 3 and 4).

Additionally, prototype windows are clamped to a maximum width of 200 ms (`max_window_ms` in `erpxttn.py`). This prevents late, broad components (e.g., LPC, LateP) from dominating the prototype representation. A future direction is to make this configurable per-dataset, as late ERP components naturally span wider temporal windows than early ones.

## Output Artifacts

For each ERPXTTN fold, the following are saved:

| File | Description |
|------|-------------|
| `results.json` | Per-subject metrics, detected windows (ERPXTTN), and run metadata (`args`, `seed`, `device`, elapsed time, aggregate stats) |
| `predictions_sub-*.npz` | Predicted probabilities and ground-truth labels |
| `curves_sub-*.npz` | Phase 1 & 2 training loss curves, validation AUROC |
| `attention_sub-*.npz` | Full attention weight tensors (B, H, N, K) per test trial |
| `prototypes_sub-*.npz` | Raw prototype waveforms and detected window boundaries |
| `fig_prototypes.png` | Prototype waveforms with detected temporal windows |
| `fig_entropy_vs_auroc.png` | Attention entropy vs. classification AUROC |
| `fig_attn_timecourse.png` | Mean attention weight timecourse (per class) |
| `fig_attn_diff_overlay.png` | Attention difference overlaid on ERP |
| `fig_per_subject_routing.png` | Per-subject prototype routing distributions |
| `fig_tp_tn_routing_sub-*_highconf.png` | High-confidence TP/TN trial attention maps |
| `fig_tp_tn_routing_sub-*_median.png` | Median-confidence TP/TN trial attention maps |

The classifier baselines (EEGNet, EEG-Deformer, EPMN, xDAWN+RG) save `results.json` and `predictions_sub-*.npz` (plus `curves_sub-*.npz` for the two-phase neural models); only ERP-XTTN additionally saves `attention_*.npz` and `prototypes_*.npz`. All artifacts live under a per-seed `seed-<N>/` directory.

## Datasets

| Dataset | Subjects | Channels | Sampling Rate | Task |
|---------|----------|----------|---------------|------|
| BNCI ErrP (013-2015) | 6 | 64 | 512 Hz → 256 | Cursor/agent observation error monitoring |
| HRI ErrP (cursor) | 11 | 27 | 256 Hz | Cursor control error monitoring |
| ERP CORE N400 | 40 | 30 | 1024 Hz → 256 | Semantic priming (related/unrelated) |
| ERP CORE ERN | 40 | 30 | 1024 Hz → 256 | Flanker task (correct/error response) |
| ERP CORE LRP | 40 | 30 | 1024 Hz → 256 | Flanker task (left/right response) |
| ERP CORE MMN | 40 | 30 | 1024 Hz → 256 | Oddball (standard/deviant) |
| ERP CORE N170 | 40 | 30 | 1024 Hz → 256 | Face/car categorization |
| ERP CORE N2pc | 40 | 30 | 1024 Hz → 256 | Visual search (target laterality) |
| ERP CORE P300 | 40 | 30 | 1024 Hz → 256 | Oddball (target/non-target) |

## Results

Result artifacts are written under `datasets/*/results/.../seed-<N>/` when the sweep is run. Use the dashboard to browse saved numbers and figures, or `python run_manager.py` to monitor / launch runs.

Per-subject LOSO results for the v1.0.0 Graz paper (BNCI + HRI, midline2 / midline3 / full) are frozen on the `v1.0.0` tag.

## Hardware

Current training runs on **NVIDIA RTX PRO 6000 Blackwell Server Edition** (96 GB VRAM) via RunPod, with 128 CPU cores and 1.5 TB RAM. Previous results (now superseded) were on an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM).

### Training configuration

- **Batch size**: 128 (increased from 32 on the old GPU — validated against old results, deltas within noise)
- **Parallel jobs**: Up to 20 GPU jobs + CPU-only xDAWN jobs concurrently
- **Thread limiting**: When running multiple jobs in parallel, set `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` to prevent CPU over-subscription. PyTorch defaults to one thread per CPU core, so with many concurrent jobs this causes severe contention and can make runs slower than a weaker GPU running serially. The `run_manager.py --daemon` mode handles this automatically.
- **Resume support**: Use `--resume` flag to skip completed folds after interruption. Prediction files are saved per-fold, so partial runs are recoverable.

### Run Manager

`run_manager.py` manages the experiment queue:

```bash
python run_manager.py                      # Status check
python run_manager.py --launch             # Launch next batch
python run_manager.py --launch-all         # Fill all GPU slots
python run_manager.py --daemon             # Loop: auto-launch as slots free up
python run_manager.py --include-full       # Include full-channel runs in queue
```

For overnight runs:
```bash
nohup python run_manager.py --daemon --include-full > logs/run_manager.log 2>&1 &
```

### Dashboard

A browser-based dashboard is available at `dashboard.html`. When running on RunPod with JupyterLab:

```
https://<POD_ID>-8888.proxy.runpod.net/files/workspace/ERP-XTTN/dashboard.html?token=<JUPYTER_TOKEN>
```

Get the token with: `ps aux | grep -oP '(?<=token=)\S+' | head -1`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code (v2.0.0), please cite the **extension paper preprint** and the **software**:

```bibtex
@misc{wyman2026erpxttn_preprint,
  title={ERP-XTTN: Interpretable Prototype-Guided Cross-Attention for Cross-Subject ERP Classification},
  author={Wyman, Charlotte Genevier and Hirshfield, Leanne},
  year={2026},
  eprint={2606.02939},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  note={Preprint. Full publication metadata will be added upon acceptance.}
}

@software{wyman2026erpxttn_code,
  author={Wyman, Charlotte Genevier},
  title={ERP-XTTN: Interpretable Cross-Attention ERP Classifier},
  year={2026},
  publisher={Zenodo},
  version={v2.0.0},
  doi={10.5281/zenodo.20497891},
  url={https://doi.org/10.5281/zenodo.20497891}
}
```

The earlier **v1.0.0 conference release** (*Graz BCI 2026*) is archived separately and available via the [`v1.0.0` release](https://github.com/cgenevier/ERP-XTTN/releases/tag/v1.0.0):

```bibtex
@inproceedings{wyman2026erpxttn,
  title={ERP-XTTN: Interpretable Cross-Subject Error-Related Potential Classification via Cross-Attention to Data-Driven ERP Prototypes},
  author={Wyman, Charlotte Genevier and Hirshfield, Leanne},
  booktitle={Proceedings of the 10th Graz Brain-Computer Interface Conference},
  year={2026},
  note={To appear. DOI and full publication metadata will be added upon publication.}
}
```

The `@software` DOI above (`10.5281/zenodo.20497891`) is specific to v2.0.0; the badge links the Zenodo concept DOI (`10.5281/zenodo.20087550`), which always resolves to the latest release. Machine-readable metadata is in [CITATION.cff](CITATION.cff).
