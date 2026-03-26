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
├── run_manager.py          # Batch launcher / status monitor for experiment runs
├── dashboard.html          # Browser dashboard for browsing saved results
├── eegnet.py               # EEGNet baseline (Lawhern et al., 2018)
├── erpxttn.py              # ERP-XTTN model + dynamic peak detection
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
│   ├── erpcore_p300/               # ERP CORE — P300
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
| **xDAWN+RG** | xDAWN spatial filtering + Riemannian geometry classifier (classical ML baseline) |
| **ERPXTTN Fixed** | Cross-attention prototype model with dataset-configured polarity pattern; results saved under `erpxttn_fixed/` |
| **ERPXTTN Auto** | Auto peak-detection variant of ERPXTTN; results saved under `erpxttn_auto/` |

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

# Fixed-pattern ERPXTTN
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn

# Auto ERPXTTN
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn --peak-mode auto

```

`--model erpxttn` selects the ERPXTTN architecture. Constrained runs are saved
under `erpxttn_fixed/`; auto runs are saved under `erpxttn_auto/`.

### 5. Generate Figures

```bash
python 05_gen_figures.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn_fixed
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
| ERP CORE P300 | 40 | 30 | 1024 Hz → 256 | Oddball (target/non-target) |

## Prototype Configuration

ERPXTTN uses data-driven prototype detection from the training-set grand-average difference wave. The following parameters are configured per dataset in `dataset_config.json`, informed by established ERP morphology:

- **Polarity pattern**: The expected alternating sign pattern of ERP components (e.g., `neg, pos` for a negative-then-positive waveform). Determines the number of prototypes (K).
- **Peak prominence**: Minimum prominence threshold (in z-scored units) for peak detection. Tuned per dataset to ensure the detector anchors on canonical ERP components rather than small early deflections.
- **Proto names**: Human-readable labels for each prototype window, corresponding to known ERP components.

Prototype windows are detected automatically per LOSO fold from the training data only — no test-set information is used. The polarity pattern and prominence threshold encode prior neuroscience knowledge about each paradigm's expected waveform morphology.

| Dataset | Pattern | K | Prominence | Prototypes |
|---------|---------|---|------------|------------|
| BNCI ErrP | pos, neg, pos, neg | 4 | 0.02 | P1-diff, Ne-diff, Pe-diff, LateN-diff |
| HRI ErrP | pos, neg, pos, neg | 4 | 0.02 | P1-diff, Ne-diff, Pe-diff, LateN-diff |
| ERN | neg, pos | 2 | 0.02 | ERN-diff, Pe-diff |
| LRP | neg, pos | 2 | 0.02 | LRP, LateP |
| MMN | neg, pos, neg | 3 | 0.02 | MMN-diff, P3a-diff, LateN-diff |
| N170 | neg, pos, neg | 3 | 0.02 | N170-diff, VPP-diff, LateN-diff |
| N2pc | neg, pos, neg | 3 | 0.02 | N2pc-diff, SPCN-diff, LateN-diff |
| N400 | pos, neg, pos, neg | 4 | 0.02 | P2-diff, N400-diff, LPC-diff, LateN-diff |
| P300 | pos, neg, pos | 3 | 0.02 | P3-diff, SW-diff, LateP-diff |

Additionally, prototype windows are clamped to a maximum width of 200ms (`max_window_ms` in `erpxttn.py`). This prevents late, broad components (e.g., LPC, LateP) from dominating the prototype representation. A future direction is to make this configurable per-dataset, as late ERP components naturally span wider temporal windows than early ones.

## Results Summary

Main result artifacts are included in the repo under `datasets/*/results/`.
Use the dashboard to browse saved numbers and figures, or `python run_manager.py`
to monitor / launch future reruns.

## Hardware

Current training runs on **NVIDIA RTX PRO 6000 Blackwell Server Edition** (98 GB VRAM) via RunPod, with 128 CPU cores and 1.5 TB RAM. Previous results (now superseded) were on an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM).

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

Please refer to the accompanying paper for citation and usage terms.
