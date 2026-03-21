# ERP-XTTN: Cross-Attention ERP Classifier with Dynamic Prototypes

Code repository for ERP-XTTN, an interpretable cross-attention architecture for ERP classification from EEG signals.

## Repository Structure

```
ERP-XTTN/
├── 01_convert_data.py      # Raw data → MNE Raw FIF + BIDS export
├── 02_inspect.py           # Data inspection / quality check utilities
├── 03_preprocess.py        # Preprocessing + epoching pipeline
├── 04_train.py             # LOSO cross-validation training
├── 05_gen_figures.py       # Attention & TP/TN figure generation
├── eegnet.py               # EEGNet baseline (Lawhern et al., 2018)
├── erpxttn.py              # ERP-XTTN model + dynamic peak detection + RCL
├── xdawn_rg.py             # xDAWN + Riemannian Geometry baseline
├── datasets/
│   ├── bnci_errp_013-2015/         # BNCI Horizon 2020 ErrP (feedback)
│   ├── hri_errp_cursor/            # HRI cursor ErrP (feedback)
│   ├── erpcore_ern/                # ERP CORE — ERN (response-locked)
│   ├── erpcore_lrp/                # ERP CORE — LRP
│   ├── erpcore_mmn/                # ERP CORE — MMN
│   ├── erpcore_n170/               # ERP CORE — N170
│   ├── erpcore_n2pc/               # ERP CORE — N2pc
│   ├── erpcore_n400/               # ERP CORE — N400
│   ├── erpcore_p3/                 # ERP CORE — P3
└── logs/
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
| **xDAWN+RG** | xDAWN spatial filtering + Riemannian geometry classifier (classical ML baseline) |
| **ERPXTTN** | Cross-attention prototype model with dynamic peak detection |
| **ERPXTTN (RCL)** | ERPXTTN with routing contrast loss — contrastive regularizer that encourages class-discriminative attention routing |

### Routing Contrast Loss (RCL)

ERPXTTN supports an optional routing contrast loss controlled by `--routing-contrast-weight`:
- **Weight = 0** (default): Standard ERPXTTN — no auxiliary loss
- **Weight > 0** (e.g., 0.3): Adds a contrastive loss that maximizes the distance between prototype routing features across classes

The RCL computes summary features from the attention map (prototype mass, confidence, entropy, diagonality, center-of-mass) and penalizes when these features are similar between positive and negative classes. This encourages the model to learn class-discriminative routing patterns without adding any learnable parameters.

Results directories are automatically suffixed: `erpxttn` for standard, `erpxttn_rcl0.3` for RCL with weight 0.3.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- MNE-Python
- NumPy, SciPy, scikit-learn, matplotlib
- pyriemann (for xDAWN+RG baseline)

```bash
pip install torch mne numpy scipy scikit-learn matplotlib pyriemann
pip install mne-bids pandas pybv  # for data conversion only
```

## Pipeline

### 1. Convert Raw Data

```bash
python 01_convert_data.py datasets/bnci_errp_013-2015
python 01_convert_data.py datasets/erpcore_n400
```

### 2. Inspect (Optional QC)

```bash
python 02_inspect.py datasets/erpcore_n400
```

### 3. Preprocess and Epoch

```bash
python 03_preprocess.py --dataset datasets/bnci_errp_013-2015 --channels midline3 --resample 256
python 03_preprocess.py --dataset datasets/erpcore_n400 --channels midline3_n400 --resample 256
python 03_preprocess.py --dataset datasets/erpcore_n400 --channels full --resample 256
```

### 4. Train (LOSO Cross-Validation)

```bash
# EEGNet baseline
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model eegnet

# xDAWN+RG classical baseline
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model xdawn_rg

# Standard ERPXTTN
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn

# ERPXTTN with routing contrast loss
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn --routing-contrast-weight 0.3
```

### 5. Generate Figures

```bash
python 05_gen_figures.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn
python 05_gen_figures.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn_rcl0.3
```

## Datasets

| Dataset | Subjects | Channels | Sampling Rate | Task |
|---------|----------|----------|---------------|------|
| BNCI ErrP (013-2015) | 6 | 64 | 512 Hz → 256 | P300 speller error monitoring |
| HRI ErrP (cursor) | 11 | 27 | 256 Hz | Cursor control error monitoring |
| ERP CORE N400 | 40 | 30 | 1024 Hz → 256 | Semantic priming (related/unrelated) |
| ERP CORE ERN | 40 | 30 | 1024 Hz → 256 | Flanker task (correct/error response) |
| ERP CORE LRP | 40 | 30 | 1024 Hz → 256 | Flanker task (left/right response) |
| ERP CORE MMN | 40 | 30 | 1024 Hz → 256 | Oddball (standard/deviant) |
| ERP CORE N170 | 40 | 30 | 1024 Hz → 256 | Face/car categorization |
| ERP CORE N2pc | 40 | 30 | 1024 Hz → 256 | Visual search (target laterality) |
| ERP CORE P3 | 40 | 30 | 1024 Hz → 256 | Oddball (target/non-target) |

## Prototype Configuration

ERPXTTN uses data-driven prototype detection from the training-set grand-average difference wave. The following parameters are configured per dataset in `dataset_config.json`, informed by established ERP morphology:

- **Polarity pattern**: The expected alternating sign pattern of ERP components (e.g., `neg, pos` for a negative-then-positive waveform). Determines the number of prototypes (K).
- **Peak prominence**: Minimum prominence threshold (in z-scored units) for peak detection. Tuned per dataset to ensure the detector anchors on canonical ERP components rather than small early deflections.
- **Proto names**: Human-readable labels for each prototype window, corresponding to known ERP components.

Prototype windows are detected automatically per LOSO fold from the training data only — no test-set information is used. The polarity pattern and prominence threshold encode prior neuroscience knowledge about each paradigm's expected waveform morphology.

| Dataset | Pattern | K | Prominence | Prototypes |
|---------|---------|---|------------|------------|
| BNCI ErrP | pos, neg, pos, neg | 4 | 0.02 | P1, Ne, Pe, LateN |
| HRI ErrP | pos, neg, pos, neg | 4 | 0.02 | P1, Ne, Pe, LateN |
| ERN | neg, pos | 2 | 0.005 | ERN, Pe |
| LRP | neg, pos, neg, pos | 4 | 0.005 | EarlyN, EarlyP, LRP, LateP |
| MMN | neg, pos, neg | 3 | 0.005 | MMN, P3a, LateN |
| N170 | neg, pos, neg | 3 | 0.005 | N170, VPP, LateN |
| N2pc | neg, pos, neg | 3 | 0.005 | N2pc, SPCN, LateN |
| N400 | pos, neg, pos, neg | 4 | 0.005 | P2, N400, LPC, LateN |
| P3 | pos, neg, pos | 3 | 0.10 | P3, SW, LateP |

Additionally, prototype windows are clamped to a maximum width of 200ms (`max_window_ms` in `erpxttn.py`). This prevents late, broad components (e.g., LPC, LateP) from dominating the prototype representation. A future direction is to make this configurable per-dataset, as late ERP components naturally span wider temporal windows than early ones.

## Results Summary


### 3-Channel Results (Mean AUROC)

| Model | BNCI ErrP | HRI ErrP | N400 |
|-------|-----------|----------|------|
| **EEGNet** | **0.806** | **0.864** | **0.582** |
| **xDAWN+RG** | 0.785 | 0.837 | 0.580 |
| **ERPXTTN** | 0.782 | 0.837 | 0.570 |
| **ERPXTTN (RCL 0.3)** | 0.782 | 0.834 | 0.565 |

### Full-Channel Results (Mean AUROC)

| Model | BNCI ErrP | HRI ErrP | N400 |
|-------|-----------|----------|------|
| **EEGNet** | **0.842** | **0.884** | **0.763** |
| **xDAWN+RG** | 0.821 | 0.869 | 0.712 |
| **ERPXTTN** | 0.805 | 0.834 | 0.725 |

## Hardware

Experiments run on NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM).

## License

Please refer to the accompanying paper for citation and usage terms.
