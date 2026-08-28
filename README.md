# ERP-XTTN: Interpretable Cross-Attention ERP Classifier

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20087550-blue)](https://doi.org/10.5281/zenodo.20087550)

ERP-XTTN is a neurophysiologically-grounded, prototype-guided cross-attention model for cross-subject event-related potential (ERP) classification from EEG. It routes peaks detected in each trial to ERP prototypes derived only from the training subjects, combines the learned routing with a fixed physiological similarity, and adds a separately grounded amplitude pathway through leakage-safe late fusion.

> **Version 3.0.0** introduces the peak-unit, grounded-readout ERP-XTTN used in the revised submission. It replaces the v2 patch-grid/free-readout model with load-bearing prototypes, native-width whitened matching, channel-resolved amplitude factors, groundedness validation, expanded baselines, and five-seed experiments across nine datasets and two montage settings.

## Versions

- **v3.0.0** *(current code)* — Grounded peak-unit routing, two-factor fusion, groundedness interventions, EEG-Deformer and EPMN baselines, full-montage experiments, and expanded ablations.
  - Paper: [Journal of Neural Engineering](https://doi.org/10.1088/1741-2552/ae9f95)
  - Archived on Zenodo: [10.5281/zenodo.21810709](https://doi.org/10.5281/zenodo.21810709)
- **[v2.0.0](https://github.com/cgenevier/ERP-XTTN/releases/tag/v2.0.0)** — ERP CORE extension using the earlier patch-grid/automatic-prototype architecture.
  - Paper: [arXiv:2606.02939](https://arxiv.org/abs/2606.02939)
  - Archived on Zenodo: [10.5281/zenodo.20497891](https://doi.org/10.5281/zenodo.20497891)
- **[v1.0.0](https://github.com/cgenevier/ERP-XTTN/releases/tag/v1.0.0)** — Graz BCI 2026 conference implementation.
  - Paper: Coming soon
  - Archived on Zenodo: [10.5281/zenodo.20087551](https://doi.org/10.5281/zenodo.20087551)

To reproduce an earlier release, check out its tag, for example:
```bash
git checkout v2.0.0
```

## Repository Structure

```text
ERP-XTTN/
├── 01_convert_data.py       # Raw data → BIDS + MNE Raw FIF
├── 02_inspect.py            # Data inspection and quality-control utilities
├── 03_preprocess.py         # Filtering, channel selection, resampling, epoching
├── 04_train.py              # LOSO training, evaluation, and two-factor fusion
├── 05_gen_figures.py        # Per-dataset routing and morphology figures
├── 06_gen_analysis.py       # Cross-dataset analysis → JSON/CSV summaries
├── 07_gen_paper_figures.py  # Paper figures and LaTeX tables
├── 08_validate.py           # Frozen-model groundedness intervention battery
├── erpxttn.py               # Grounded peak-unit ERP-XTTN and Stage-2 fusion
├── ablation_erpxttn.py      # End-to-end, no-whitening, and free-head ablations
├── ablation_amp_only.py     # Amplitude-only Stage-2 ablation
├── ablation_val_learned.py  # Groundedness contrast for the free-head ablation
├── eegnet.py                # EEGNet baseline
├── eeg_deformer.py          # EEG-Deformer baseline
├── epmn.py                  # ERP Prototypical Matching Network baseline
├── xdawn_rg.py              # xDAWN + Riemannian Geometry baseline
├── bench_latency.py         # Parameter-count and inference-latency benchmark
├── run_manager.py           # Experiment queue, launcher, and status monitor
├── dashboard.html           # Browser dashboard for archived results
├── analysis_summary.json    # Cached output from 06_gen_analysis.py
├── stats_summary.csv        # Paired statistical results
├── paper_figures/           # Current paper figures, tables, and morphology cache
│   └── graz2026/            # Frozen figures from the v1 conference release
└── datasets/
    └── <dataset>/
        ├── dataset_config.json
        ├── original_data/    # Not tracked
        ├── epoched_fif/      # Not tracked
        └── results/          # Archived predictions, routing, analyses, figures
```

## Models

| Model | Description |
|---|---|
| **ERP-XTTN** | Main v3 model: peak-unit QK routing, fixed whitened-cosine readout, and channel-resolved amplitude late fusion. Training uses `--model erpxttn`; results are stored under `erpxttn_peak/`. |
| **EEGNet** | Compact convolutional baseline following Lawhern et al. (2018). |
| **EEG-Deformer** | Dense convolutional transformer baseline following Ding et al. (2025). |
| **EPMN** | ERP Prototypical Matching Network following Wei et al. (2022), evaluated with the matched shared protocol and a native episodic robustness recipe. |
| **xDAWN+RG** | Deterministic xDAWN spatial filtering plus Riemannian tangent-space logistic regression. |
| **ERP-XTTN ablations** | No whitening, end-to-end head, learned free readout, no self-attention, K/prominence/head-count changes, routing-only, and amplitude-only variants. |

Neural models are evaluated over seeds 1–5. xDAWN+RG is deterministic and uses seed 1. Performance statistics average each subject across available seeds before paired testing; heavier routing/prototype analyses use seed 1 as the reference seed.

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended
- The archived experiments used Python 3.11.14 and an NVIDIA RTX PRO 6000 Blackwell Server Edition on RunPod.

Install the pinned RunPod environment:
```bash
pip install -r requirements.txt
```
The conversion step also requires:
```bash
pip install mne-bids pybv
```
The provided requirements select the CUDA 12.8 PyTorch wheel. For a CPU-only environment, replace the CUDA-specific PyTorch source and pin with an appropriate CPU build.

## Reproducing the Experiments

### 1. Acquire the data

- **BNCI Horizon 2020 013-2015 ErrP** — 6 subjects, 64 EEG channels, 512 Hz. Download from the [BNCI database](https://bnci-horizon-2020.eu/database/data-sets) and place the source files in `datasets/bnci_errp_013-2015/original_data/`.
- **HRI Cursor ErrP** — 11 subjects, 27 EEG channels, 256 Hz. Download from the [dataset repository](https://github.com/stefan-ehrlich/dataset-ErrP-HRI/) and place the EEGLAB files in `datasets/hri_errp_cursor/original_data/`.
- **ERP CORE** — 40 subjects, seven paradigms, 30 analyzed EEG channels, 1024 Hz. Download from [OSF](https://osf.io/thsqg/) and place each paradigm in `datasets/erpcore_<paradigm>/original_data/`.

Each dataset's `dataset_config.json` defines its source format, subjects, events, labels, channel presets, detection channel, preprocessing variants, and prototype prominence threshold.

### 2. Convert to BIDS and FIF
```bash
python 01_convert_data.py datasets/bnci_errp_013-2015
python 01_convert_data.py datasets/hri_errp_cursor
python 01_convert_data.py datasets/erpcore_n400
```
Repeat the ERP CORE command for ERN, LRP, MMN, N170, N2pc, N400, and P300.

### 3. Inspect the converted data
```bash
python 02_inspect.py datasets/erpcore_n400
```

### 4. Preprocess and epoch

The archived variants use no re-reference, a forward IIR bandpass, 256 Hz resampling where needed, a 0–800 ms epoch, and no QC-based epoch dropping.
```bash
# ErrP examples
python 03_preprocess.py --dataset datasets/bnci_errp_013-2015 --channels midline3 --resample 256 --no-qc-drop
python 03_preprocess.py --dataset datasets/hri_errp_cursor --channels midline3 --resample 256 --no-qc-drop

# ERP CORE example: paradigm-specific 3-channel and full-montage variants
python 03_preprocess.py --dataset datasets/erpcore_n400 --channels midline3_n400 --resample 256 --no-qc-drop
python 03_preprocess.py --dataset datasets/erpcore_n400 --channels full --resample 256 --no-qc-drop
```

### 5. Train a single LOSO run
```bash
# Main grounded ERP-XTTN; writes to erpxttn_peak/seed-1/
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model erpxttn --seed 1

# Baselines
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model eegnet --seed 1
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model eeg_deformer --seed 1
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model epmn --seed 1
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model xdawn_rg --seed 1

# Native-recipe EPMN robustness run
python 04_train.py --dataset erpcore_n400 --channels midline3_n400 --model epmn --epmn-recipe native --seed 1
```
Use `--resume` to skip folds that already have prediction files.

### 6. Run the full experiment matrix

The paper uses the following three-channel presets and also evaluates each dataset with `--channels full`:

| Dataset | Three-channel preset | Prototype detection channel |
|---|---|---|
| `bnci_errp_013-2015` | `midline3` | Cz |
| `hri_errp_cursor` | `midline3` | Cz |
| `erpcore_ern` | `midline3_ern` | FCz |
| `erpcore_lrp` | `lateral3_lrp` | C3 |
| `erpcore_mmn` | `midline3` | FCz |
| `erpcore_n170` | `occipital3_n170` | PO8 |
| `erpcore_n2pc` | `posterior3_n2pc` | PO7 |
| `erpcore_n400` | `midline3_n400` | CPz |
| `erpcore_p300` | `midline3` | Pz |
```bash
# Status only
python run_manager.py --include-full --include-ablation --include-epmn-native

# Fill all available slots
python run_manager.py --launch-all --include-full --include-ablation --include-epmn-native

# Continuously refill slots until the queue is complete
python run_manager.py --daemon --include-full --include-ablation --include-epmn-native
```
The complete managed queue contains 608 runs: five seeds for each neural model, one xDAWN+RG run per dataset/montage, the four-dataset ERP-XTTN ablation grid, and native-EPMN robustness runs on ERN, P300, and N400.

Representative ablation commands are:
```bash
python 04_train.py --dataset hri_errp_cursor --channels midline3 --model ablation_erpxttn --ablation-mode e2e --ablation-tag e2e --seed 1
python 04_train.py --dataset hri_errp_cursor --channels midline3 --model ablation_erpxttn --ablation-mode nowhiten --ablation-tag nowhiten --seed 1
python 04_train.py --dataset hri_errp_cursor --channels midline3 --model ablation_erpxttn --ablation-mode learned_readout --ablation-tag learned_readout --seed 1
python 04_train.py --dataset hri_errp_cursor --channels midline3 --model erpxttn --no-self-attn --ablation-tag nosa --seed 1
python 04_train.py --dataset hri_errp_cursor --channels midline3 --model erpxttn --max-k 2 --ablation-tag k2 --seed 1
python ablation_amp_only.py --dataset hri_errp_cursor --seed 1
```

### 7. Generate analyses, figures, and tables
```bash
# Per-dataset peak-routing decompositions and morphology panels
python 05_gen_figures.py --dataset erpcore_n400 --channels midline3_n400
python 05_gen_figures.py --dataset erpcore_n400 --channels midline3_n400 --morphology-only

# Cross-dataset metrics and paired statistics
python 06_gen_analysis.py

# Publication figures and LaTeX tables
python 07_gen_paper_figures.py
```
`06_gen_analysis.py` writes `analysis_summary.json` and `stats_summary.csv`. `07_gen_paper_figures.py` writes the current manuscript assets directly into `paper_figures/`.

### 8. Run the groundedness validation
```bash
# Frozen-model intervention battery for one dataset and seed
python 08_validate.py --dataset erpcore_p300 --channels midline3 --seed 1

# Synthetic end-to-end smoke test
python 08_validate.py --self-test

# Free-head contrast; omitting arguments loops over available ablation datasets/seeds
python ablation_val_learned.py --dataset erpcore_p300 --seed 1
```

The validation battery measures contribution concentration, causal occlusion, template-swap ladders, carrier scrambles, component deletion, amplitude localization, and the combined routing/amplitude decision. The learned-readout contrast shows what changes when the classifier is allowed to ignore the grounded match.

## ERP-XTTN v3 Architecture

### 1. Fold-specific ERP prototypes

For each LOSO fold and training phase, the model computes the positive-minus-negative grand-average difference wave using training subjects only. On the configured detection channel it finds up to K=4 prominent positive or negative deflections after 50 ms, separated by at least 80 ms. Prototype windows expand to neighboring zero crossings and are clamped to 40–200 ms.

Each prototype retains three representations:

- a compact multichannel segment resampled to eight samples for its routing key;
- a native-width, jointly spatiotemporally whitened template for physiological matching;
- a raw full-window template for amplitude projections.

K is fold-dependent. In the archived results it is 4 for BNCI, HRI, N170, N2pc, and P300; 3 for ERN; 2 for LRP; and 3–4 for MMN and N400.

### 2. Per-trial peak units

Each trial is smoothed on its detection channel and tokenized into up to 14 prominent positive or negative peaks. Peaks are separated by at least 40 ms and bounded by their flanking inflection points. Each variable-width peak segment is:

- resampled to eight samples for the learned embedding; and
- retained at native width for the fixed whitened-cosine match to every prototype.

This distinction is important: `patch_width=8` controls only the learned embedding width, not the physiological comparison window.

### 3. Learned peak-to-prototype routing

Peak embeddings receive interpolated positional encodings and masked self-attention. Multi-head QK cross-attention then routes each valid trial peak to the fold's prototypes, producing `a[b,p,k]`. Prototypes are keys; there is no learned value projection in the main routing block.

### 4. Fixed grounded routing readout

For peak p and prototype k, `m[b,p,k]` is the native-width whitened-cosine match. It is computed directly from the signal and frozen training-fold templates and carries no gradient.

The routing logit is:
```text
routing_logit = scale × Σ(a[p,k] × m[p,k]) / number_of_valid_peaks + bias
```
Only the routing attention, an overall scale, and a bias can learn. Because the decision explicitly multiplies attention by physiological match, prototypes are load-bearing by construction.

### 5. Grounded amplitude factor and leakage-safe fusion

The model also projects each trial onto each prototype template, retaining per-channel matched-filter terms and all pairwise bipolar channel contrasts. A logistic Stage-2 combiner fuses:
```text
[routing_logit, channel-resolved matched filters, bipolar contrasts]
```
For held-out subject s, features for both combiner training and testing are produced by s's frozen fold model, which was trained without s. The combiner is fit only on the other subjects and then applied to s. No held-out labels, calibration trials, or another fold's model enter s's prediction.

## Training Procedure

- **Evaluation:** leave-one-subject-out cross-validation.
- **Phase 1:** 85% training and 15% subject/label-stratified validation; early stopping patience 15, maximum 250 epochs.
- **Phase 2:** reinitialize and train on the complete non-test pool for `best_epoch + 1` epochs.
- **Prototype construction:** rerun on the phase-specific training pool; test-subject data are never used.
- **Optimizer:** AdamW, learning rate 1e-3, weight decay 1e-4.
- **Schedule:** five-epoch linear warmup and cosine decay on a 100-epoch schedule.
- **Loss:** class-weighted binary cross-entropy.
- **Augmentation:** temporal jitter up to ±10 samples plus Gaussian noise with σ=0.1.
- **Normalization:** per-channel training-pool z-score applied unchanged to validation/test data.
- **Gradient clipping:** maximum norm 1.0.
- **Batch size:** 128.
- **Paper seeds:** 1, 2, 3, 4, and 5. The direct CLI default is 42 for ad hoc runs.

## Output Artifacts

For a main ERP-XTTN run, artifacts are stored under:
```text
datasets/<dataset>/results/tmin0ms_tmax800ms/<variant>/erpxttn_peak/seed-<N>/
```
| File | Contents |
|---|---|
| `results.json` | Arguments, fold metrics, routing-only summary, and final two-factor summary. |
| `predictions_sub-*.npz` | Routing probabilities, labels, metrics, routing logits, and scalar matched-filter values. |
| `curves_sub-*.npz` | Phase 1/2 loss and validation curves. |
| `routing_sub-*.npz` | Grounded attention a, match m, valid-peak masks, centers, bounds, normalized test trials, templates, and metadata. |
| `prototypes_sub-*.npz` | Compact prototype segments, raw templates, native windows, and sampling rate. |
| `two_factor_sub-*.npz` | Final fused probabilities, coefficients, feature names/slices, labels, and AUROC. |
| `checkpoint_sub-*.pt` | Frozen fold model and normalization used by validation; generated locally but excluded from git as regenerable. |
| `validation*.json` | Groundedness validation and learned-readout contrasts where available. |
| `fig_*.png` | Per-subject routing decompositions and morphology figures. |

Baseline directories use the same per-seed layout and contain their applicable `results.json`, predictions, and training curves. The tracked result artifacts, cross-dataset summaries, dashboard, figures, and tables are included in the Zenodo archive.

## Datasets

| Dataset | Subjects | Recorded/analyzed channels | Sampling rate | Task |
|---|---:|---:|---:|---|
| BNCI ErrP 013-2015 | 6 | 64 | 512 → 256 Hz | Feedback error monitoring |
| HRI Cursor ErrP | 11 | 27 | 256 Hz | Robot/cursor action error monitoring |
| ERP CORE ERN | 40 | 30 analyzed | 1024 → 256 Hz | Flanker response accuracy |
| ERP CORE LRP | 40 | 30 analyzed | 1024 → 256 Hz | Flanker response laterality |
| ERP CORE MMN | 40 | 30 analyzed | 1024 → 256 Hz | Auditory oddball |
| ERP CORE N170 | 40 | 30 analyzed | 1024 → 256 Hz | Face/car categorization |
| ERP CORE N2pc | 40 | 30 analyzed | 1024 → 256 Hz | Visual-search target laterality |
| ERP CORE N400 | 40 | 30 analyzed | 1024 → 256 Hz | Semantic relatedness |
| ERP CORE P300 | 40 | 30 analyzed | 1024 → 256 Hz | Visual oddball |

## Results and Hardware

The repository includes the result artifacts used for the revised submission. Use `dashboard.html` to browse archived values and figures, or run `run_manager.py` without launch flags to audit experiment completeness.

The v3 experiments ran on an NVIDIA RTX PRO 6000 Blackwell Server Edition with 96 GB VRAM via RunPod, with up to 20 concurrent GPU jobs. `run_manager.py` sets `OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8` for launched processes to prevent CPU oversubscription.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).

## Citation

If you use this code, please cite both the paper and the archived software:
```bibtex
@article{wyman2026erpxttn,
  author={Wyman, Charlotte Genevier and Hirshfield, Leanne},
  title={ERP-XTTN: Interpretable Prototype-Guided Cross-Attention
         for Cross-Subject ERP Classification},
  journal={Journal of Neural Engineering},
  year={2026},
  doi={10.1088/1741-2552/ae9f95},
  url={https://doi.org/10.1088/1741-2552/ae9f95}
}

@software{wyman2026erpxttn_code,
  author={Wyman, Charlotte Genevier},
  title={ERP-XTTN: Interpretable Cross-Attention ERP Classifier},
  year={2026},
  publisher={Zenodo},
  version={v3.0.0},
  doi={10.5281/zenodo.21810709},
  url={https://doi.org/10.5281/zenodo.21810709}
}
```

The earlier **v2.0.0 extension release** is archived separately and available via the [`v2.0.0` release](https://github.com/cgenevier/ERP-XTTN/releases/tag/v2.0.0). For that release, cite both the extension-paper preprint and the archived software:
```bibtex
@misc{wyman2026erpxttn_preprint,
  title={ERP-XTTN: Interpretable Prototype-Guided Cross-Attention for Cross-Subject ERP Classification},
  author={Wyman, Charlotte Genevier and Hirshfield, Leanne},
  year={2026},
  doi={10.48550/arXiv.2606.02939},
  eprint={2606.02939},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  note={Preprint of the paper published in Journal of Neural Engineering,
        \url{https://doi.org/10.1088/1741-2552/ae9f95}.}
}

@software{wyman2026erpxttn_code_v2,
  author={Wyman, Charlotte Genevier},
  title={ERP-XTTN: Interpretable Cross-Attention ERP Classifier},
  year={2026},
  publisher={Zenodo},
  version={v2.0.0},
  doi={10.5281/zenodo.20497891},
  url={https://doi.org/10.5281/zenodo.20497891}
}
```

The earlier **v1.0.0 conference release** (*Graz BCI 2026*) is archived separately and available via the [`v1.0.0` release](https://github.com/cgenevier/ERP-XTTN/releases/tag/v1.0.0). For that release, cite both the conference paper and the archived software:
```bibtex
@inproceedings{wyman2026erpxttn,
  title={ERP-XTTN: Interpretable Cross-Subject Error-Related Potential Classification via Cross-Attention to Data-Driven ERP Prototypes},
  author={Wyman, Charlotte Genevier and Hirshfield, Leanne},
  booktitle={Proceedings of the 10th Graz Brain-Computer Interface Conference},
  year={2026},
  note={To appear. DOI and full publication metadata will be added upon publication.}
}

@software{wyman2026erpxttn_code_v1,
  author={Wyman, Charlotte Genevier},
  title={ERP-XTTN: Interpretable Cross-Attention ERP Classifier},
  year={2026},
  publisher={Zenodo},
  version={v1.0.0},
  doi={10.5281/zenodo.20087551},
  url={https://doi.org/10.5281/zenodo.20087551}
}
```

Machine-readable citation metadata are in [CITATION.cff](CITATION.cff).
