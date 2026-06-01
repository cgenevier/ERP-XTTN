#!/usr/bin/env python3
"""06_gen_analysis.py — cross-dataset interpretability analysis.

Computes a battery of metrics per (dataset, variant) for the paper model
`erpxttn_auto`, plus paired-LOSO interpretability tax against EEGNet and
xDAWN+RG baselines. Output is a single JSON file consumed by the
Analysis tab in `dashboard.html`.

Metrics per (dataset, variant):
    - Paired interpretability tax vs EEGNet and vs xDAWN+RG
        (mean subject-paired AUROC delta, SEM, Wilcoxon p)
    - Minority class proportion
    - K statistics: mode, range, consistency (fraction at mode)
    - Prototype stability (positional, per-slot mean pairwise Pearson r
        at detection channel, averaged across slots)
    - Grand-average diff-wave SNR proxy at each prototype's peak
    - Mean normalized attention entropy across subjects
    - Routing discriminability (cosine distance between class-averaged
        attention vectors)
    - Per-subject AUROC SD
    - Latency variability (SD of peak latency of dominant component
        across per-subject difference waves)
    - Pearson TP↔FP and TP↔TN correlation at detection channel, with
        bootstrap 95% CI across subjects
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import mne
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
OUTPUT_PATH = REPO_ROOT / "analysis_summary.json"

# Only process the paper model.
MODEL = "erpxttn_auto"
BASELINES = ["eegnet", "xdawn_rg"]

N_BOOTSTRAP = 1000
RNG_SEED = 42


# =====================================================================
# Helpers (lift from 05_gen_figures patterns)
# =====================================================================

def discover_combos():
    """Return list of (dataset_dir, variant_key, variant_dir, cfg) for
    every dataset × variant that has an erpxttn_auto results directory."""
    combos = []
    for dcfg in sorted(DATASETS_DIR.glob("*/dataset_config.json")):
        cfg = json.load(open(dcfg))
        cfg.setdefault("name", dcfg.parent.name)
        for vk, vd in sorted(cfg.get("variants", {}).items()):
            mp = dcfg.parent / "results" / "tmin0ms_tmax800ms" / vd / MODEL
            if (mp / "results.json").exists():
                combos.append((dcfg.parent.name, vk, vd, cfg))
    return combos


def load_subject_epochs(cfg, channel_config, subject):
    """Load held-out subject's raw epochs and channel names."""
    variant = cfg["variants"][channel_config]
    base = DATASETS_DIR / cfg["name"] / "epoched_fif" / "tmin0ms_tmax800ms" / variant
    pos_key = cfg["label_map"]["pos_key"]
    neg_key = cfg["label_map"]["neg_key"]
    label_groups = cfg.get("label_groups")
    subj_dir = base / subject
    all_X, all_y, ch_names = [], [], None
    for fif in sorted(subj_dir.rglob("*-epo.fif")):
        epochs = mne.read_epochs(str(fif), preload=True, verbose=False)
        if ch_names is None:
            ch_names = epochs.ch_names
        evid = epochs.event_id
        if label_groups:
            pos_names = set(label_groups.get(pos_key, []))
            neg_names = set(label_groups.get(neg_key, []))
            pos_ids = {v for k, v in evid.items() if k in pos_names}
            neg_ids = {v for k, v in evid.items() if k in neg_names}
        else:
            pos_ids = {v for k, v in evid.items() if pos_key in k.lower()}
            neg_ids = {v for k, v in evid.items() if neg_key in k.lower()}
        keep_ids = pos_ids | neg_ids
        if not keep_ids:
            continue
        X = epochs.get_data()
        codes = epochs.events[:, 2]
        mask = np.isin(codes, list(keep_ids))
        all_X.append(X[mask])
        y = np.array([1 if c in pos_ids else 0 for c in codes[mask]])
        all_y.append(y)
    if not all_X:
        return None, None, None
    X = np.concatenate(all_X, 0).astype(np.float32)
    y = np.concatenate(all_y, 0).astype(np.int64)
    return X, y, ch_names


def detection_channel_index(cfg, ch_names):
    """Resolve detection channel name to index in the loaded ch_names."""
    det = cfg.get("detection_channel", "Cz")
    if det in ch_names:
        return ch_names.index(det), det
    # Fallback: second channel (existing convention in generate code)
    fallback = min(1, len(ch_names) - 1)
    return fallback, ch_names[fallback]


def load_results_json(dataset_dir, variant_dir, model):
    p = DATASETS_DIR / dataset_dir / "results" / "tmin0ms_tmax800ms" / variant_dir / model / "results.json"
    if not p.exists():
        return None
    return json.load(open(p))


def per_subject_auroc_map(dataset_dir, variant_dir, model):
    """{subject: test_auroc} from results.json if present."""
    r = load_results_json(dataset_dir, variant_dir, model)
    if not r:
        return {}
    return {f["test_subject"]: float(f["test_auroc"]) for f in r.get("folds", [])}


def pearson_or_nan(a, b):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    r, _ = stats.pearsonr(a, b)
    return float(r)


def bootstrap_mean_ci(values, n=N_BOOTSTRAP, seed=RNG_SEED):
    """95% percentile bootstrap CI on the mean."""
    arr = np.asarray([v for v in values if np.isfinite(v)])
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        idx = rng.integers(0, arr.size, arr.size)
        means.append(arr[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(arr.mean()), float(lo), float(hi))


# =====================================================================
# Per-metric computation
# =====================================================================

def metric_tax_paired(ds_dir, var_dir):
    """Paired-subject AUROC delta: baseline - erpxttn_auto.

    Uses subjects that appear in both models' results.json.
    Returns {baseline: {mean, sem, n, wilcoxon_p}} per baseline.
    """
    erp = per_subject_auroc_map(ds_dir, var_dir, MODEL)
    out = {}
    for bl in BASELINES:
        base = per_subject_auroc_map(ds_dir, var_dir, bl)
        common = sorted(set(erp) & set(base))
        if not common:
            out[bl] = None
            continue
        deltas = np.array([base[s] - erp[s] for s in common])
        mean = float(deltas.mean())
        sem = float(deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
        try:
            _, p = stats.wilcoxon(deltas) if len(deltas) > 1 and np.any(deltas != 0) else (None, None)
            p = float(p) if p is not None else None
        except Exception:
            p = None
        out[bl] = {"mean_delta": mean, "sem": sem, "n_subjects": len(common),
                   "wilcoxon_p": p, "per_subject": [float(d) for d in deltas]}
    return out


def metric_class_balance(ds_dir, var_dir):
    """Minority-class proportion pooled across all subjects' test sets."""
    results_dir = DATASETS_DIR / ds_dir / "results" / "tmin0ms_tmax800ms" / var_dir / MODEL
    all_y = []
    for p in sorted(results_dir.glob("predictions_*.npz")):
        d = np.load(p)
        all_y.append(d["labels"])
    if not all_y:
        return None
    y = np.concatenate(all_y)
    p1 = float((y == 1).mean())
    return min(p1, 1 - p1)


def metric_k_stats(ds_dir, var_dir):
    results_dir = DATASETS_DIR / ds_dir / "results" / "tmin0ms_tmax800ms" / var_dir / MODEL
    Ks = []
    for p in sorted(results_dir.glob("prototypes_*.npz")):
        Ks.append(int(np.load(p)["proto_raw"].shape[0]))
    if not Ks:
        return None
    counts = {k: Ks.count(k) for k in set(Ks)}
    mode_k = max(counts, key=counts.get)
    return {
        "K_values": Ks,
        "K_mode": mode_k,
        "K_min": min(Ks),
        "K_max": max(Ks),
        "K_consistency": counts[mode_k] / len(Ks),
        "n_folds": len(Ks),
    }


def metric_proto_stability(ds_dir, var_dir, cfg):
    """Positional per-slot mean pairwise Pearson r across folds at
    detection channel, averaged across slots (slot-k only contributes if
    at least 2 folds have it)."""
    results_dir = DATASETS_DIR / ds_dir / "results" / "tmin0ms_tmax800ms" / var_dir / MODEL
    proto_files = sorted(results_dir.glob("prototypes_*.npz"))
    if len(proto_files) < 2:
        return None
    # Need channel ordering to locate detection channel. Channel names
    # come from subject epoch files.
    first_subj = proto_files[0].stem.replace("prototypes_", "")
    _, _, ch_names = load_subject_epochs(cfg, [v for v in cfg["variants"] if cfg["variants"][v] == var_dir][0], first_subj)
    if ch_names is None:
        return None
    det_idx, det_name = detection_channel_index(cfg, ch_names)

    # Stack per-fold: list of (K_f, T) arrays at detection channel.
    # Each prototype is zero outside its window, so we correlate the
    # FULL T-length signal — this preserves peak latency alignment
    # across folds (essential: a peak at 250 ms in fold A vs 280 ms in
    # fold B should correlate lower than two peaks at 255 ms).
    fold_data = []
    fold_windows = []
    for p in proto_files:
        d = np.load(p)
        proto = d["proto_raw"]  # (K, C, T)
        windows = d["proto_windows_ms"]  # (K, 2)
        fold_data.append(proto[:, det_idx, :])
        fold_windows.append(windows)
    K_max = max(a.shape[0] for a in fold_data)
    per_slot_scores = []
    per_slot_n = []
    for k in range(K_max):
        sigs = []
        for fd, fw in zip(fold_data, fold_windows):
            if fd.shape[0] <= k:
                continue
            # Full-length slot-k signal (zero-padded outside window).
            # Restrict correlation to the UNION of all contributing
            # folds' windows so we compare over the actual signal
            # support and not a mostly-zero vector.
            sigs.append(fd[k])
        if len(sigs) < 2:
            per_slot_scores.append(np.nan)
            per_slot_n.append(len(sigs))
            continue
        # Build a mask over samples where at least one fold has a
        # nonzero value (i.e., union of windows for this slot).
        stacked = np.stack(sigs, axis=0)  # (n_folds_with_slot, T)
        support = np.any(stacked != 0, axis=0)
        if support.sum() < 3:
            per_slot_scores.append(np.nan)
            per_slot_n.append(len(sigs))
            continue
        region = stacked[:, support]
        pair_rs = []
        for a, b in combinations(region, 2):
            pair_rs.append(pearson_or_nan(a, b))
        pair_rs = [r for r in pair_rs if np.isfinite(r)]
        per_slot_scores.append(float(np.mean(pair_rs)) if pair_rs else np.nan)
        per_slot_n.append(len(sigs))
    valid = [s for s in per_slot_scores if np.isfinite(s)]
    overall = float(np.mean(valid)) if valid else np.nan
    return {
        "detection_channel": det_name,
        "per_slot_pairwise_r": per_slot_scores,
        "per_slot_n_folds": per_slot_n,
        "mean_pairwise_r": overall,
    }


def metric_snr_and_ga_diff(ds_dir, var_key, cfg, loaded_epochs_cache):
    """For each fold's training-subject pool, approximate SNR as
    peak-to-peak GA diff wave divided by trial-to-trial SD at the peak
    latency. Computed at detection channel, averaged across folds.

    Uses the diff-wave peak latencies already stored per fold
    (`proto_windows_ms` centers). Per-trial SD at peak is computed
    over all trials (both classes combined) in the training pool.
    """
    var_dir = cfg["variants"][var_key]
    results_dir = DATASETS_DIR / ds_dir / "results" / "tmin0ms_tmax800ms" / var_dir / MODEL
    subjects = [r["test_subject"] for r in load_results_json(ds_dir, var_dir, MODEL)["folds"]]
    if not subjects:
        return None
    # Preload all subjects' epochs (channel ordering shared)
    _get = loaded_epochs_cache
    # detection channel
    X0, y0, ch_names = _get(subjects[0])
    if X0 is None:
        return None
    det_idx, det_name = detection_channel_index(cfg, ch_names)

    per_fold_snrs = []
    for test_subj in subjects:
        proto_path = results_dir / f"prototypes_{test_subj}.npz"
        if not proto_path.exists():
            continue
        pdata = np.load(proto_path)
        windows_ms = pdata["proto_windows_ms"]
        sfreq = float(pdata["sfreq"])
        # Build training pool (all subjects except test_subj)
        pool_X, pool_y = [], []
        for s in subjects:
            if s == test_subj:
                continue
            Xs, ys, _ = _get(s)
            if Xs is None:
                continue
            pool_X.append(Xs); pool_y.append(ys)
        if not pool_X:
            continue
        pool_X = np.concatenate(pool_X, 0)
        pool_y = np.concatenate(pool_y, 0)
        # Grand-average diff wave at detection channel
        err = pool_X[pool_y == 1, det_idx, :].mean(0) if (pool_y == 1).any() else None
        cor = pool_X[pool_y == 0, det_idx, :].mean(0) if (pool_y == 0).any() else None
        if err is None or cor is None:
            continue
        diff = err - cor
        # Trial-to-trial SD across all trials at each time point
        sd_t = pool_X[:, det_idx, :].std(axis=0, ddof=1)
        per_slot = []
        for (s_ms, e_ms) in windows_ms:
            s_samp = int(round(s_ms / 1000.0 * sfreq))
            e_samp = int(round(e_ms / 1000.0 * sfreq))
            s_samp = max(0, min(s_samp, len(diff) - 1))
            e_samp = max(s_samp + 1, min(e_samp, len(diff)))
            seg = diff[s_samp:e_samp]
            if seg.size < 2:
                continue
            peak_off = int(np.argmax(np.abs(seg)))
            peak_samp = s_samp + peak_off
            peak_amp = abs(float(diff[peak_samp]))
            noise = float(sd_t[peak_samp])
            if noise > 0:
                per_slot.append(peak_amp / noise)
        if per_slot:
            per_fold_snrs.append(float(np.mean(per_slot)))
    if not per_fold_snrs:
        return None
    return {
        "detection_channel": det_name,
        "mean_snr": float(np.mean(per_fold_snrs)),
        "sd_snr": float(np.std(per_fold_snrs, ddof=1)) if len(per_fold_snrs) > 1 else 0.0,
        "n_folds": len(per_fold_snrs),
    }


def metric_attention_entropy(ds_dir, var_dir):
    """Mean normalized attention entropy across subjects.
    Follows the entropy definition used in 05_gen_figures for
    fig_entropy_vs_auroc: per-trial normalized over the flattened
    (N_patches × K) attention vector, then mean across trials and
    subjects.
    """
    results_dir = DATASETS_DIR / ds_dir / "results" / "tmin0ms_tmax800ms" / var_dir / MODEL
    per_subj = []
    for p in sorted(results_dir.glob("attention_*.npz")):
        attn = np.load(p)["attention_weights"]  # (B, H, N, K)
        attn_mean = attn.mean(axis=1)  # (B, N, K)
        flat = attn_mean.reshape(attn_mean.shape[0], -1)
        norm = flat / (flat.sum(1, keepdims=True) + 1e-10)
        ent = -np.sum(norm * np.log(norm + 1e-10), axis=1)
        norm_ent = ent / np.log(norm.shape[1])
        per_subj.append(float(norm_ent.mean()))
    if not per_subj:
        return None
    return {
        "per_subject_entropy": per_subj,
        "mean_entropy": float(np.mean(per_subj)),
        "sd_entropy": float(np.std(per_subj, ddof=1)) if len(per_subj) > 1 else 0.0,
    }


def metric_routing_discriminability(ds_dir, var_dir):
    """Cosine distance between class-averaged attention vectors
    (flatten over N_patches × K). Computed per subject, then averaged.
    """
    results_dir = DATASETS_DIR / ds_dir / "results" / "tmin0ms_tmax800ms" / var_dir / MODEL
    per_subj = []
    for p in sorted(results_dir.glob("attention_*.npz")):
        subj = p.stem.replace("attention_", "")
        attn = np.load(p)["attention_weights"].mean(axis=1)  # (B, N, K)
        labs = np.load(p)["labels"]
        if (labs == 1).sum() == 0 or (labs == 0).sum() == 0:
            continue
        err_vec = attn[labs == 1].mean(axis=0).flatten()
        cor_vec = attn[labs == 0].mean(axis=0).flatten()
        # Cosine distance = 1 - cos_similarity
        num = float(np.dot(err_vec, cor_vec))
        den = float(np.linalg.norm(err_vec) * np.linalg.norm(cor_vec))
        cos_sim = num / den if den > 0 else 0.0
        per_subj.append(1.0 - cos_sim)
    if not per_subj:
        return None
    return {
        "per_subject_distance": per_subj,
        "mean_distance": float(np.mean(per_subj)),
        "sd_distance": float(np.std(per_subj, ddof=1)) if len(per_subj) > 1 else 0.0,
    }


def metric_auroc_sd(ds_dir, var_dir):
    r = load_results_json(ds_dir, var_dir, MODEL)
    if not r:
        return None
    aurocs = [f["test_auroc"] for f in r["folds"]]
    return {
        "mean": float(np.mean(aurocs)),
        "sd": float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
        "n_subjects": len(aurocs),
    }


def metric_latency_variability(ds_dir, var_key, cfg, loaded_epochs_cache):
    """SD (across subjects) of the peak latency of the dominant
    difference-wave component at the detection channel. Dominant =
    largest absolute smoothed peak after MIN_P1_LATENCY_MS.

    Requires per-subject class means; uses the subject's own epochs
    (not the LOSO training pool) so each fold contributes one latency
    estimate.
    """
    MIN_LATENCY_MS = 50.0
    _get = loaded_epochs_cache
    # Probe first subject to get channel names
    subjects = [r["test_subject"] for r in load_results_json(ds_dir, cfg["variants"][var_key], MODEL)["folds"]]
    if not subjects:
        return None
    X0, y0, ch_names = _get(subjects[0])
    if X0 is None:
        return None
    det_idx, det_name = detection_channel_index(cfg, ch_names)
    # Assume sfreq = 256 (repo convention)
    sfreq = 256.0
    min_latency_samp = int(round(MIN_LATENCY_MS / 1000.0 * sfreq))

    peak_latencies_ms = []
    for s in subjects:
        Xs, ys, _ = _get(s)
        if Xs is None:
            continue
        if (ys == 1).sum() == 0 or (ys == 0).sum() == 0:
            continue
        diff = Xs[ys == 1, det_idx, :].mean(0) - Xs[ys == 0, det_idx, :].mean(0)
        smoothed = gaussian_filter1d(diff, sigma=2.0)
        # Dominant |peak| after min latency
        idx = min_latency_samp + int(np.argmax(np.abs(smoothed[min_latency_samp:])))
        peak_latencies_ms.append(idx / sfreq * 1000.0)
    if not peak_latencies_ms:
        return None
    return {
        "per_subject_latency_ms": peak_latencies_ms,
        "sd_latency_ms": float(np.std(peak_latencies_ms, ddof=1)) if len(peak_latencies_ms) > 1 else 0.0,
        "mean_latency_ms": float(np.mean(peak_latencies_ms)),
        "n_subjects": len(peak_latencies_ms),
    }


def metric_tp_fp_tn_correlations(ds_dir, var_key, cfg, loaded_epochs_cache):
    """Per subject: Pearson r between mean TP waveform and mean FP
    waveform (and TP vs TN) at detection channel. Aggregate with
    bootstrap 95% CI across subjects.
    """
    var_dir = cfg["variants"][var_key]
    results_dir = DATASETS_DIR / ds_dir / "results" / "tmin0ms_tmax800ms" / var_dir / MODEL
    subjects = [r["test_subject"] for r in load_results_json(ds_dir, var_dir, MODEL)["folds"]]
    if not subjects:
        return None
    _get = loaded_epochs_cache
    X0, y0, ch_names = _get(subjects[0])
    if X0 is None:
        return None
    det_idx, det_name = detection_channel_index(cfg, ch_names)

    tp_fp, tp_tn = [], []
    n_skipped = 0
    for s in subjects:
        pred_path = results_dir / f"predictions_{s}.npz"
        if not pred_path.exists():
            n_skipped += 1
            continue
        d = np.load(pred_path)
        probs, labels = d["probs"], d["labels"]
        Xs, ys, _ = _get(s)
        if Xs is None or len(probs) != Xs.shape[0]:
            n_skipped += 1
            continue
        preds = (probs >= 0.5).astype(int)
        Xd = Xs[:, det_idx, :]
        tp_mask = (labels == 1) & (preds == 1)
        fn_mask = (labels == 1) & (preds == 0)
        tn_mask = (labels == 0) & (preds == 0)
        fp_mask = (labels == 0) & (preds == 1)
        if tp_mask.sum() == 0:
            continue
        tp_mean = Xd[tp_mask].mean(axis=0)
        if fp_mask.sum() > 0:
            fp_mean = Xd[fp_mask].mean(axis=0)
            tp_fp.append(pearson_or_nan(tp_mean, fp_mean))
        if tn_mask.sum() > 0:
            tn_mean = Xd[tn_mask].mean(axis=0)
            tp_tn.append(pearson_or_nan(tp_mean, tn_mean))
    tp_fp_m, tp_fp_lo, tp_fp_hi = bootstrap_mean_ci(tp_fp)
    tp_tn_m, tp_tn_lo, tp_tn_hi = bootstrap_mean_ci(tp_tn)
    return {
        "detection_channel": det_name,
        "tp_fp": {"per_subject_r": tp_fp, "mean_r": tp_fp_m,
                  "ci_low": tp_fp_lo, "ci_high": tp_fp_hi, "n_subjects": len(tp_fp)},
        "tp_tn": {"per_subject_r": tp_tn, "mean_r": tp_tn_m,
                  "ci_low": tp_tn_lo, "ci_high": tp_tn_hi, "n_subjects": len(tp_tn)},
        "n_skipped": n_skipped,
    }


# =====================================================================
# Main
# =====================================================================

def compute_for_combo(ds_dir, var_key, var_dir, cfg):
    """Compute all metrics for one (dataset, variant). Epochs cache is
    built once and shared across metrics that need raw signals."""
    print(f"\n=== {ds_dir} / {var_key} ===", flush=True)
    entry = {
        "dataset_dir": ds_dir,
        "dataset_name": cfg.get("dataset_name", ds_dir),
        "variant_key": var_key,
        "variant_dir": var_dir,
        "detection_channel": cfg.get("detection_channel", "Cz"),
        "paradigm_class_labels": {
            "pos_key": cfg.get("label_map", {}).get("pos_key"),
            "neg_key": cfg.get("label_map", {}).get("neg_key"),
        },
    }

    print("  tax (paired)...", flush=True)
    entry["tax"] = metric_tax_paired(ds_dir, var_dir)

    print("  class balance...", flush=True)
    entry["class_balance_minority"] = metric_class_balance(ds_dir, var_dir)

    print("  K stats...", flush=True)
    entry["k_stats"] = metric_k_stats(ds_dir, var_dir)

    print("  prototype stability...", flush=True)
    entry["proto_stability"] = metric_proto_stability(ds_dir, var_dir, cfg)

    print("  attention entropy...", flush=True)
    entry["attention_entropy"] = metric_attention_entropy(ds_dir, var_dir)

    print("  routing discriminability...", flush=True)
    entry["routing_discriminability"] = metric_routing_discriminability(ds_dir, var_dir)

    print("  AUROC SD...", flush=True)
    entry["auroc"] = metric_auroc_sd(ds_dir, var_dir)

    # Metrics needing raw epochs (shared cache across them)
    print("  loading subject epochs (shared cache)...", flush=True)
    cache = {}

    def _get(subj):
        if subj not in cache:
            cache[subj] = load_subject_epochs(cfg, var_key, subj)
        return cache[subj]

    print("  SNR proxy...", flush=True)
    entry["snr_proxy"] = metric_snr_and_ga_diff(ds_dir, var_key, cfg, _get)

    print("  latency variability...", flush=True)
    entry["latency_variability"] = metric_latency_variability(ds_dir, var_key, cfg, _get)

    print("  TP-FP / TP-TN correlations...", flush=True)
    entry["tp_fp_tn_corr"] = metric_tp_fp_tn_correlations(ds_dir, var_key, cfg, _get)

    # Release the epoch cache memory before moving on
    cache.clear()

    return entry


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset analysis")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Where to write analysis_summary.json")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of 'dataset|variant' to limit to (for debugging)")
    parser.add_argument("--variant", choices=["3ch", "full", "all"], default="all",
                        help="Filter variants by class (default: all)")
    args = parser.parse_args()

    combos = discover_combos()
    if args.only:
        picks = set(args.only.split(","))
        combos = [c for c in combos if f"{c[0]}|{c[1]}" in picks]
    if args.variant == "3ch":
        combos = [c for c in combos if c[1] != "full"]
    elif args.variant == "full":
        combos = [c for c in combos if c[1] == "full"]

    print(f"Computing analysis for {len(combos)} (dataset, variant) combos.", flush=True)

    results = {}
    for ds_dir, var_key, var_dir, cfg in combos:
        key = f"{ds_dir}|{var_key}"
        try:
            results[key] = compute_for_combo(ds_dir, var_key, var_dir, cfg)
        except Exception as e:
            import traceback
            print(f"FAILED {key}: {e}", flush=True)
            traceback.print_exc()
            results[key] = {"error": str(e)}

    out = {
        "model": MODEL,
        "baselines": BASELINES,
        "n_combos": len(results),
        "combos": results,
        "metric_descriptions": {
            "tax": "Paired LOSO interpretability tax: per-subject (baseline_AUROC − ERPXTTN_Auto_AUROC), averaged. Positive = baseline wins. Includes SEM and Wilcoxon signed-rank p.",
            "class_balance_minority": "Proportion of epochs belonging to the minority class, pooled across all test subjects. 0.5 = perfect balance.",
            "k_stats": "Number of prototypes (K) detected by auto peak-finder per LOSO fold. K_mode is the most common value; K_consistency is the fraction of folds at K_mode.",
            "proto_stability": "Positional prototype stability: per-prototype-slot mean pairwise Pearson r across LOSO folds at the detection channel, averaged across slots. Higher = prototypes look more alike across folds. Note: the metric treats slot-index as canonical, so it mixes shape stability with K-consistency (see K_stats for the latter alone).",
            "snr_proxy": "Grand-average difference-wave SNR at each prototype's peak latency, per fold: peak |amplitude| ÷ trial-to-trial SD across all training-pool trials at that same sample. Averaged across prototypes within a fold and across folds. Higher = stereotyped component against noisier trial-level signal.",
            "attention_entropy": "Mean normalized attention entropy across subjects. Computed per trial over the flattened (N_patches × K) attention vector and normalized by log(N_patches × K). 0 = peaked routing (one patch×prototype dominates), 1 = diffuse.",
            "routing_discriminability": "Mean cosine distance between class-averaged attention vectors (error vs correct) across subjects. Higher = more class-discriminative routing.",
            "auroc": "LOSO test AUROC for erpxttn_auto. `sd` is cross-subject standard deviation (proxy for generalization consistency).",
            "latency_variability": "Cross-subject SD (ms) of the peak latency of the dominant difference-wave component at the detection channel (per-subject diff wave, smoothed, argmax of |amplitude| after 50 ms). Large SD = the component moves around across subjects.",
            "tp_fp_tn_corr": "Pearson r between a subject's TP grand-mean waveform and their FP grand-mean waveform at the detection channel (and TP↔TN as contrast). Aggregated across subjects as mean ± 95% bootstrap CI. TP↔FP ≫ TP↔TN ⇒ the model's false alarms morphologically resemble its hits, which is a signature of waveform-based classification.",
        },
    }

    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}  ({args.output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
