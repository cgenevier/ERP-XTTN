# ERP-XTTN: Interpretable Cross-Attention ERP Classifier

Code repository for ERP-XTTN, an interpretable cross-attention architecture for ERP classification from EEG signals. Currently applied to error-related potentials (ErrPs); see Versions for ongoing work.

## Versions

- **v1.0.0** (this version) — Code for Wyman & Hirshfield, *Graz BCI 2026*.
  - Paper DOI: [pending]
  - Archived on Zenodo: [10.5281/zenodo.20087551](https://doi.org/10.5281/zenodo.20087551)
- **v2.0.0** *(planned)* — Extension paper in preparation, covering the full
  ERP CORE benchmark with additional baselines. See the `erp-core` branch
  for work in progress.

To reproduce the Graz paper results, check out the `v1.0.0` tag after release:
`git checkout v1.0.0`

## Repository Structure

```
ERP-XTTN/
├── 01_convert_data.py      # Raw data → MNE Raw FIF + BIDS export
├── 02_inspect.py           # Data inspection / quality check utilities
├── 03_preprocess.py        # Preprocessing + epoching pipeline
├── 04_train.py             # LOSO cross-validation training (EEGNet & ERP-XTTN)
├── eegnet.py               # EEGNet baseline (Lawhern et al., 2018)
├── erpxttn.py              # ERP-XTTN model + dynamic peak detection
├── 05_gen_figures.py        # Attention & TP/TN figure generation
├── datasets/
│   ├── bnci_horizon_2020_ErrP/
│   │   ├── original_data/           # Raw .mat files (not tracked in git)
│   │   ├── bids/                    # BIDS-formatted EEG (not tracked)
│   │   ├── epoched_fif/             # Preprocessed epochs (not tracked)
│   │   │   └── tmin0ms_tmax800ms/
│   │   │       ├── noref_midline2_rs256_iir_fwd_bp-1-10/
│   │   │       ├── noref_midline3_rs256_iir_fwd_bp-1-10/
│   │   │       └── noref_rs256_iir_fwd_bp-1-10/          # full (64ch)
│   │   └── results/
│   │       └── tmin0ms_tmax800ms/
│   │           └── <variant>/<model>/
│   │               ├── results.json
│   │               ├── predictions_sub-*.npz
│   │               ├── curves_sub-*.npz
│   │               ├── attention_sub-*.npz      # ERP-XTTN only
│   │               ├── prototypes_sub-*.npz     # ERP-XTTN only
│   │               └── fig_*.png                # ERP-XTTN only
│   └── hri_cursor/
│       └── (same structure as above, without rs256 in variant names)
└── logs/
    └── <timestamp>_<dataset>_<channels>_<model>/
        └── train.log
```

## Requirements

- Python 3.11+ (Tested with Python 3.11.14)
- CUDA-capable GPU recommended (experiments used an NVIDIA RTX 4070)

Install dependencies:

```bash
pip install -r requirements.txt
```

For data conversion only (step 1), additional packages:
```bash
pip install mne-bids pybv
```

## Steps to Reproduce

### 1. Data Acquisition

**BNCI Horizon 2020 — Monitoring Error-Related Potentials** (Margaux et al.)
- 6 subjects, 64 EEG channels, 512 Hz
- Publicly available at https://bnci-horizon-2020.eu/database/data-sets
- Place raw `.mat` files in `datasets/bnci_horizon_2020_ErrP/original_data/`

**HRI Cursor** (Ehrlich & Cheng, 2019)
- 11 subjects (sub-02 through sub-13, excluding sub-01 and sub-12), 27 EEG channels, 256 Hz
- Publicly available at https://github.com/stefan-ehrlich/dataset-ErrP-HRI/
- Place raw `.set/.fdt` files in `datasets/hri_cursor/original_data/`

Each dataset directory requires a `dataset_config.json` specifying format, event IDs, channel info, and epoching rules. Training subject lists and result variant names are defined in `04_train.py` and `05_gen_figures.py`.

### 2. Convert to BIDS + FIF

```bash
python 01_convert_data.py datasets/bnci_horizon_2020_ErrP
python 01_convert_data.py datasets/hri_cursor
```

### 3. Preprocess and Epoch

Preprocessing applies: optional re-referencing, channel selection, IIR 1–10 Hz bandpass (4th-order Butterworth, forward-phase), optional resampling, and epoching (0–800 ms post-event). The commands below include `--no-qc-drop` to reproduce the variant names used by the checked-in results and training scripts.

```bash
# BNCI — midline subsets (resampled 512→256 Hz)
python 03_preprocess.py --dataset datasets/bnci_horizon_2020_ErrP --channels midline2 --reference none --resample 256 --no-qc-drop
python 03_preprocess.py --dataset datasets/bnci_horizon_2020_ErrP --channels midline3 --reference none --resample 256 --no-qc-drop
python 03_preprocess.py --dataset datasets/bnci_horizon_2020_ErrP --channels full    --reference none --resample 256 --no-qc-drop

# HRI — no resampling needed (already 256 Hz)
python 03_preprocess.py --dataset datasets/hri_cursor --channels midline2 --reference none --no-qc-drop
python 03_preprocess.py --dataset datasets/hri_cursor --channels midline3 --reference none --no-qc-drop
python 03_preprocess.py --dataset datasets/hri_cursor --channels full    --reference none --no-qc-drop
```

### 4. Train Models (LOSO Cross-Validation)

```bash
# Example: train both models on BNCI midline3
python 04_train.py --dataset bnci --channels midline3 --model eegnet
python 04_train.py --dataset bnci --channels midline3 --model erpxttn

# Full grid (2 datasets × 3 channel configs × 2 models = 12 runs)
for dataset in bnci hri; do
  for channels in midline2 midline3 full; do
    for model in eegnet erpxttn; do
      python 04_train.py --dataset $dataset --channels $channels --model $model
    done
  done
done
```

Results are saved to `datasets/<name>/results/tmin0ms_tmax800ms/<variant>/<model>/results.json`.

### 5. Generate Figures (ERP-XTTN only)

```bash
# Run one configuration
python 05_gen_figures.py --dataset bnci --channels midline3

# Generate all figure sets (2 datasets × 3 channel configs)
for dataset in bnci hri; do
  for channels in midline2 midline3 full; do
    python 05_gen_figures.py --dataset $dataset --channels $channels
  done
done
```

Generates per-subject attention routing figures (TP/TN high-confidence and median trials) and aggregate attention analysis plots (prototype visualization, entropy vs. AUROC, attention timecourse, differential overlay, per-subject routing).

## Architecture

### ERP-XTTN

ERP-XTTN classifies EEG trials by cross-attending input signal patches against ERP prototype templates derived from the grand-average difference wave. The architecture has five stages:

**1. Patch Embedding + Positional Encoding**

The input epoch (C channels × T time samples) is divided into N = T / w non-overlapping temporal patches of width w = 8 samples (31.25 ms at 256 Hz). Each patch is flattened to a C·w-dimensional vector and linearly projected to d = 64 dimensions. Learned positional embeddings are added to encode temporal position.

**2. Self-Attention**

A single multi-head self-attention layer (H = 4 heads, d_h = 16) with pre-layer-normalization and residual connection refines the patch embeddings, allowing patches to share temporal context before cross-attention.

**3. Prototype Construction (per fold, per training phase)**

For each LOSO fold, K = 4 ERP prototypes are extracted from the grand-average difference wave (error minus correct) on the Cz detection channel. This is done once in Phase 1 (train split) and again in Phase 2 (full training pool):

- The difference wave is Gaussian-smoothed (σ = 2 samples), and local extrema with prominence ≥ 0.1 are considered. P1 is anchored as the earliest positive peak at least 50 ms post-event; the remaining Ne–Pe–LateN chain (negative–positive–negative) is selected to maximize total prominence.
- Window boundaries expand from each peak to neighboring zero-crossings, clamped to [40, 200] ms width.
- Each prototype is the segment of the full multichannel difference wave within its detected window, zero-padded elsewhere to epoch length.

Prototypes are embedded through the same patch embedding layer as input patches and mean-pooled across the patch dimension to produce K vectors of dimension d. Shared positional encoding (indexed at each prototype's temporal center) is added, placing prototypes in the same positional space as input patches.

**4. QK-Only Cross-Attention**

Input patch embeddings serve as queries (Q) and prototype embeddings as keys (K). Separate layer norms are applied before Q and K linear projections. Scaled dot-product attention produces weights in (B, H, N, K) — no value (V) projection is used. The attention map directly encodes how each input patch routes to each prototype.

**5. Classification**

Attention weights are averaged across heads, flattened to N·K dimensions, and passed through a single linear layer to produce a scalar logit.

### EEGNet (Baseline)

EEGNet (Lawhern et al., 2018) is used as a compact CNN baseline. It applies temporal convolution, depthwise spatial convolution, separable convolution, and average pooling, with max-norm weight constraints. Implementation follows Table 2 of the original paper with F1=8, D=2, F2=16, dropout=0.25.

### Training Procedure

- **Cross-validation**: Leave-one-subject-out (LOSO)
- **Two-phase training per fold**:
  - Phase 1: Train on 85% of pooled leave-out data with 15% stratified validation split. Early stopping (patience=15) on validation AUROC determines best epoch count.
  - Phase 2: Retrain from scratch on the full training pool for exactly best_epoch+1 epochs. Evaluate on the held-out test subject.
  - ERP-XTTN prototype extraction: Run in both phases, using the phase-specific training data.
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **LR schedule**: Linear warmup (5 epochs, from lr/10) + cosine annealing over 100 epochs (to lr/100)
- **Loss**: BCEWithLogitsLoss with inverse class-frequency pos_weight
- **Augmentation** (training only): Temporal jitter (uniform ±10 samples) + Gaussian noise (σ=0.1)
- **Normalization**: Per-channel z-score computed on training data, applied to train/val/test
- **Gradient clipping**: Max norm = 1.0
- **Batch size**: 32
- **Seed**: 42 (deterministic)

### Parameter Counts

| Config | Channels | EEGNet | ERP-XTTN |
|--------|----------|--------|----------|
| BNCI midline2 | 2 | 2,001 | 28,133 |
| BNCI midline3 | 3 | 2,017 | 28,645 |
| BNCI full | 64 | 2,993 | 59,877 |
| HRI midline2 | 2 | 2,001 | 28,133 |
| HRI midline3 | 3 | 2,017 | 28,645 |
| HRI full | 27 | 2,401 | 40,933 |

## Results

### Summary

| Dataset | Channels | EEGNet AUROC | ERP-XTTN AUROC | EEGNet BA | ERP-XTTN BA |
|---------|----------|-------------|---------------|----------|------------|
| BNCI | midline2 (2ch) | 0.7773 ± 0.0818 | 0.7749 ± 0.0765 | 0.7035 ± 0.0650 | 0.6984 ± 0.0624 |
| BNCI | midline3 (3ch) | 0.8060 ± 0.0767 | 0.7820 ± 0.0577 | 0.7221 ± 0.0638 | 0.6870 ± 0.0579 |
| BNCI | full (64ch) | 0.8415 ± 0.0714 | 0.8053 ± 0.0601 | 0.7564 ± 0.0839 | 0.7194 ± 0.0562 |
| HRI | midline2 (2ch) | 0.8394 ± 0.1045 | 0.8173 ± 0.1152 | 0.7624 ± 0.1038 | 0.7386 ± 0.1024 |
| HRI | midline3 (3ch) | 0.8637 ± 0.0906 | 0.8365 ± 0.1074 | 0.7785 ± 0.0912 | 0.7592 ± 0.1001 |
| HRI | full (27ch) | 0.8840 ± 0.0734 | 0.8343 ± 0.1001 | 0.7840 ± 0.0819 | 0.7468 ± 0.0890 |

### Per-Subject Results — BNCI Horizon 2020

#### midline2 (Fz, Cz)

| Subject | EEGNet AUROC | ERP-XTTN AUROC | EEGNet BA | ERP-XTTN BA |
|---------|-------------|---------------|----------|------------|
| sub-01 | 0.9093 | 0.8894 | 0.8127 | 0.7995 |
| sub-02 | 0.7755 | 0.7253 | 0.7154 | 0.6352 |
| sub-03 | 0.7897 | 0.8297 | 0.7221 | 0.7473 |
| sub-04 | 0.7229 | 0.6877 | 0.6483 | 0.6358 |
| sub-05 | 0.8219 | 0.8232 | 0.7171 | 0.7220 |
| sub-06 | 0.6447 | 0.6939 | 0.6052 | 0.6507 |
| **Mean** | **0.7773 ± 0.0818** | **0.7749 ± 0.0765** | **0.7035 ± 0.0650** | **0.6984 ± 0.0624** |

#### midline3 (Fz, Cz, Pz)

| Subject | EEGNet AUROC | ERP-XTTN AUROC | EEGNet BA | ERP-XTTN BA |
|---------|-------------|---------------|----------|------------|
| sub-01 | 0.9219 | 0.8422 | 0.8171 | 0.7569 |
| sub-02 | 0.7949 | 0.7590 | 0.7238 | 0.6580 |
| sub-03 | 0.8467 | 0.8339 | 0.7681 | 0.7016 |
| sub-04 | 0.7690 | 0.7189 | 0.6804 | 0.6058 |
| sub-05 | 0.8319 | 0.8352 | 0.7282 | 0.7602 |
| sub-06 | 0.6718 | 0.7024 | 0.6148 | 0.6396 |
| **Mean** | **0.8060 ± 0.0767** | **0.7820 ± 0.0577** | **0.7221 ± 0.0638** | **0.6870 ± 0.0579** |

#### full (64 channels)

| Subject | EEGNet AUROC | ERP-XTTN AUROC | EEGNet BA | ERP-XTTN BA |
|---------|-------------|---------------|----------|------------|
| sub-01 | 0.8570 | 0.8478 | 0.7849 | 0.7387 |
| sub-02 | 0.8922 | 0.8485 | 0.8178 | 0.7649 |
| sub-03 | 0.9168 | 0.8595 | 0.8394 | 0.7778 |
| sub-04 | 0.8077 | 0.7578 | 0.7083 | 0.6693 |
| sub-05 | 0.8746 | 0.8241 | 0.7956 | 0.7461 |
| sub-06 | 0.7007 | 0.6938 | 0.5924 | 0.6199 |
| **Mean** | **0.8415 ± 0.0714** | **0.8053 ± 0.0601** | **0.7564 ± 0.0839** | **0.7194 ± 0.0562** |

### Per-Subject Results — HRI Cursor

#### midline2 (Fz, Cz)

| Subject | EEGNet AUROC | ERP-XTTN AUROC | EEGNet BA | ERP-XTTN BA |
|---------|-------------|---------------|----------|------------|
| sub-02 | 0.7721 | 0.7173 | 0.6484 | 0.6300 |
| sub-03 | 0.8549 | 0.8406 | 0.7617 | 0.7412 |
| sub-04 | 0.9805 | 0.9699 | 0.9511 | 0.9218 |
| sub-05 | 0.7464 | 0.7332 | 0.6808 | 0.6834 |
| sub-06 | 0.8691 | 0.8520 | 0.7958 | 0.7374 |
| sub-07 | 0.8726 | 0.8390 | 0.7462 | 0.7444 |
| sub-08 | 0.8472 | 0.8366 | 0.7488 | 0.7622 |
| sub-09 | 0.8932 | 0.8806 | 0.8195 | 0.7892 |
| sub-10 | 0.5743 | 0.5278 | 0.5514 | 0.5090 |
| sub-11 | 0.8868 | 0.8616 | 0.8142 | 0.7660 |
| sub-13 | 0.9363 | 0.9320 | 0.8683 | 0.8403 |
| **Mean** | **0.8394 ± 0.1045** | **0.8173 ± 0.1152** | **0.7624 ± 0.1038** | **0.7386 ± 0.1024** |

#### midline3 (Fz, Cz, Pz)

| Subject | EEGNet AUROC | ERP-XTTN AUROC | EEGNet BA | ERP-XTTN BA |
|---------|-------------|---------------|----------|------------|
| sub-02 | 0.8276 | 0.7967 | 0.7033 | 0.6846 |
| sub-03 | 0.8675 | 0.8568 | 0.7824 | 0.7347 |
| sub-04 | 0.9862 | 0.9797 | 0.9577 | 0.9581 |
| sub-05 | 0.7665 | 0.7487 | 0.6809 | 0.6895 |
| sub-06 | 0.9066 | 0.8987 | 0.8284 | 0.8361 |
| sub-07 | 0.8796 | 0.8640 | 0.7375 | 0.7672 |
| sub-08 | 0.8793 | 0.8661 | 0.7968 | 0.7981 |
| sub-09 | 0.9231 | 0.8954 | 0.8080 | 0.7843 |
| sub-10 | 0.6357 | 0.5463 | 0.6016 | 0.5345 |
| sub-11 | 0.8857 | 0.8583 | 0.8072 | 0.7831 |
| sub-13 | 0.9428 | 0.8908 | 0.8600 | 0.7808 |
| **Mean** | **0.8637 ± 0.0906** | **0.8365 ± 0.1074** | **0.7785 ± 0.0912** | **0.7592 ± 0.1001** |

#### full (27 channels)

| Subject | EEGNet AUROC | ERP-XTTN AUROC | EEGNet BA | ERP-XTTN BA |
|---------|-------------|---------------|----------|------------|
| sub-02 | 0.8555 | 0.7075 | 0.7363 | 0.5974 |
| sub-03 | 0.8257 | 0.8118 | 0.7329 | 0.7235 |
| sub-04 | 0.9802 | 0.9583 | 0.9363 | 0.8708 |
| sub-05 | 0.8091 | 0.7471 | 0.7206 | 0.6780 |
| sub-06 | 0.9246 | 0.8508 | 0.7993 | 0.7552 |
| sub-07 | 0.9495 | 0.9230 | 0.7906 | 0.7453 |
| sub-08 | 0.8980 | 0.8296 | 0.8102 | 0.7596 |
| sub-09 | 0.9671 | 0.9397 | 0.9120 | 0.8581 |
| sub-10 | 0.7298 | 0.6239 | 0.6365 | 0.5972 |
| sub-11 | 0.8524 | 0.8803 | 0.7396 | 0.8122 |
| sub-13 | 0.9324 | 0.9050 | 0.8101 | 0.8172 |
| **Mean** | **0.8840 ± 0.0734** | **0.8343 ± 0.1001** | **0.7840 ± 0.0819** | **0.7468 ± 0.0890** |

## Output Artifacts

For each ERP-XTTN fold, the following are saved:

| File | Description |
|------|-------------|
| `results.json` | Per-subject metrics, detected windows (ERP-XTTN), and run metadata (`args`, `seed`, `device`, elapsed time, aggregate stats) |
| `predictions_sub-*.npz` | Predicted probabilities and ground-truth labels |
| `curves_sub-*.npz` | Phase 1 & 2 training loss curves, validation AUROC |
| `attention_sub-*.npz` | Full attention weight tensors (B, H, N, K) per test trial |
| `prototypes_sub-*.npz` | Raw prototype waveforms and detected window boundaries |
| `fig_prototypes.png` | Prototype waveforms with detected temporal windows |
| `fig_entropy_vs_auroc.png` | Attention entropy vs. classification AUROC |
| `fig_attn_timecourse.png` | Mean attention weight timecourse (error vs. correct) |
| `fig_attn_diff_overlay.png` | Attention difference (error - correct) overlaid on ERP |
| `fig_per_subject_routing.png` | Per-subject prototype routing distributions |
| `fig_tp_tn_routing_sub-*_highconf.png` | High-confidence TP/TN trial attention maps |
| `fig_tp_tn_routing_sub-*_median.png` | Median-confidence TP/TN trial attention maps |

## Datasets

| Dataset | Subjects | Channels | Sampling Rate | Epoch Window | Task |
|---------|----------|----------|---------------|--------------|------|
| BNCI Horizon 2020 ErrP | 6 | 64 | 512 Hz (resampled to 256) | 0–800 ms | Cursor/agent observation error monitoring |
| HRI Cursor | 11 | 27 | 256 Hz | 0–800 ms | Cursor control error monitoring |

**Channel montages evaluated**:
- `midline2`: Fz, Cz (2 channels)
- `midline3`: Fz, Cz, Pz (3 channels)
- `full`: All available channels (64 for BNCI, 27 for HRI)

## Hardware

Experiments were run on an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM). Full-channel LOSO runs complete in approximately 40–85 minutes per dataset/model combination; midline configurations are faster.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wyman2026erpxttn,
  title={ERP-XTTN: Interpretable Cross-Subject Error-Related Potential Classification via Cross-Attention to Data-Driven ERP Prototypes},
  author={Wyman, Charlotte Genevier and Hirshfield, Leanne},
  booktitle={Proceedings of the 10th Graz Brain-Computer Interface Conference},
  year={2026},
  note={To appear. DOI and full publication metadata will be added upon publication.}
}

@software{wyman2026erpxttn_code,
  author={Wyman, Charlotte Genevier},
  title={ERP-XTTN: Interpretable Cross-Attention ERP Classifier},
  year={2026},
  publisher={Zenodo},
  version={v1.0.0},
  doi={10.5281/zenodo.20087551},
  url={https://doi.org/10.5281/zenodo.20087551}
}
```

