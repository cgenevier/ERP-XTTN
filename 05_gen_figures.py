"""Generate all attention analysis and routing figures for ERP-XTTN.

Usage:
    python gen_figures.py --dataset bnci --channels midline3
    python gen_figures.py --dataset hri  --channels midline3

Generates:
    - fig_prototypes.png          Prototype waveforms across folds
    - fig_entropy_vs_auroc.png    Attention entropy vs AUROC scatter
    - fig_attn_timecourse.png     Per-prototype attention time course
    - fig_attn_diff_overlay.png   Attention difference overlay
    - fig_per_subject_routing.png Per-subject routing difference
    - fig_tp_tn_routing_*.png     TP vs TN routing per subject (high + median conf)
"""

import argparse
import json
import os
from pathlib import Path

import mne
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

PROTO_COLOR_PALETTE = ['#e67e22', '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#1abc9c']

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"


def _discover_datasets() -> list[str]:
    """Scan datasets/ for directories containing dataset_config.json."""
    return sorted(d.name for d in DATASETS_DIR.iterdir()
                  if d.is_dir() and (d / "dataset_config.json").exists())


def load_dataset_config(dataset_key: str) -> dict:
    """Load dataset config from JSON."""
    dataset_dir = DATASETS_DIR / dataset_key
    cfg_path = dataset_dir / "dataset_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No dataset_config.json found in {dataset_dir}. "
            f"Available: {_discover_datasets()}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.setdefault("name", dataset_dir.name)
    return cfg


def get_proto_config(cfg: dict, results_dir=None):
    """Return (proto_names, proto_colors) from dataset config or prototype data.

    For auto-mode results, generates P1/N1/P2/N2 names from prototype polarity.
    """
    # If results_dir provided, try to infer names from prototype data
    if results_dir is not None:
        results_dir = Path(results_dir)
        # Check if this is an auto-mode run by looking at the dir name
        if '_auto' in results_dir.name:
            # Load prototype file with the most prototypes (K varies across folds)
            proto_files = sorted(results_dir.glob("prototypes_sub-*.npz"))
            if proto_files:
                best_file, best_K = None, 0
                for pf in proto_files:
                    pK = np.load(pf)['proto_raw'].shape[0]
                    if pK > best_K:
                        best_K, best_file = pK, pf
                p = np.load(best_file)
                proto_raw = p['proto_raw']  # (K, C, T)
                windows = p['proto_windows_ms']
                K = proto_raw.shape[0]
                # Determine polarity from detection channel signal within each window
                det_ch = cfg.get('detection_channel', 'Cz')
                ch_names_cfg = None
                try:
                    ch_names_cfg = get_channel_names(cfg, list(cfg['variants'].keys())[0])
                except:
                    pass
                det_idx = 1
                if ch_names_cfg and det_ch in ch_names_cfg:
                    det_idx = ch_names_cfg.index(det_ch)

                pos_count, neg_count = 0, 0
                names = []
                for k in range(K):
                    s_samp = int(round(windows[k][0] / 1000 * 256))
                    e_samp = int(round(windows[k][1] / 1000 * 256))
                    segment = proto_raw[k, det_idx, s_samp:e_samp]
                    if len(segment) > 0:
                        peak_idx = int(np.argmax(np.abs(segment)))
                        peak_val = segment[peak_idx]
                    else:
                        peak_val = 0
                    if peak_val >= 0:
                        pos_count += 1
                        names.append(f'P{pos_count}')
                    else:
                        neg_count += 1
                        names.append(f'N{neg_count}')
                colors = PROTO_COLOR_PALETTE[:K]
                return names, colors

    names = cfg.get("proto_names", [f"Proto-{i}" for i in range(len(cfg.get("polarity_pattern", [])))])
    colors = PROTO_COLOR_PALETTE[:len(names)]
    return names, colors


def get_channel_names(cfg: dict, channel_config: str) -> list[str]:
    """Read channel names from the first epoch file."""
    variant = cfg["variants"][channel_config]
    base = DATASETS_DIR / cfg["name"] / "epoched_fif" / "tmin0ms_tmax800ms" / variant
    first_fif = next((base / cfg["subjects"][0]).rglob("*-epo.fif"))
    return mne.read_epochs(str(first_fif), preload=False, verbose=False).ch_names


def load_subject_epochs(cfg: dict, channel_config: str,
                        subject: str) -> tuple[np.ndarray, list[str]]:
    """Load raw epoch data for a subject. Returns (X_raw, ch_names)."""
    variant = cfg["variants"][channel_config]
    base = DATASETS_DIR / cfg["name"] / "epoched_fif" / "tmin0ms_tmax800ms" / variant

    pos_key = cfg["label_map"]["pos_key"]
    neg_key = cfg["label_map"]["neg_key"]
    label_groups = cfg.get("label_groups")

    all_X = []
    subj_dir = base / subject
    ch_names = None
    for fif_path in sorted(subj_dir.rglob("*-epo.fif")):
        epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
        if ch_names is None:
            ch_names = epochs.ch_names
        event_id = epochs.event_id
        if label_groups:
            pos_names = set(label_groups.get(pos_key, []))
            neg_names = set(label_groups.get(neg_key, []))
            pos_ids = {v for k, v in event_id.items() if k in pos_names}
            neg_ids = {v for k, v in event_id.items() if k in neg_names}
        else:
            pos_ids = {v for k, v in event_id.items() if pos_key in k.lower()}
            neg_ids = {v for k, v in event_id.items() if neg_key in k.lower()}
        keep_ids = pos_ids | neg_ids
        if not keep_ids:
            continue
        X = epochs.get_data()
        event_codes = epochs.events[:, 2]
        mask = np.isin(event_codes, list(keep_ids))
        all_X.append(X[mask])

    return np.concatenate(all_X, axis=0).astype(np.float32), ch_names


# =====================================================================
# Attention analysis figures (aggregate across folds)
# =====================================================================

def generate_attention_figures(results_dir, dataset_label, channels, sfreq):
    """Generate attention analysis figures.

    Args:
        results_dir: path to ERP-XTTN results directory
        dataset_label: display label for figure titles
        channels: list of channel names
        sfreq: sampling frequency in Hz
    """
    results_dir = Path(results_dir)
    with open(results_dir / 'results.json') as f:
        results = json.load(f)

    subjects = [r['test_subject'] for r in results['folds']]
    aurocs = {r['test_subject']: r['test_auroc'] for r in results['folds']}
    n_subj = len(subjects)

    all_attn, all_labels, all_protos = {}, {}, {}
    per_fold_windows = {}
    for subj in subjects:
        d = np.load(results_dir / f'attention_{subj}.npz')
        all_attn[subj] = d['attention_weights']
        all_labels[subj] = d['labels']
        p = np.load(results_dir / f'prototypes_{subj}.npz')
        all_protos[subj] = p['proto_raw']
        if 'proto_windows_ms' in p:
            per_fold_windows[subj] = [tuple(w) for w in p['proto_windows_ms']]

    # Compute mean windows across folds for aggregate figures
    # Handle variable K across folds (auto mode may detect different peak counts)
    fold_K = {s: len(per_fold_windows[s]) for s in subjects}
    K = max(fold_K.values())
    windows = []
    for k in range(K):
        k_windows = [per_fold_windows[s][k] for s in subjects if fold_K[s] > k]
        mean_win = np.round(np.mean(k_windows, axis=0), 1)
        windows.append(tuple(mean_win))
    n_with_max_K = sum(1 for v in fold_K.values() if v == K)
    if n_with_max_K < n_subj:
        print(f'  Note: {n_subj - n_with_max_K}/{n_subj} folds have fewer than {K} prototypes')
    print(f'  Dynamic windows (mean across folds): {windows}')
    C = len(channels)
    N_patches = all_attn[subjects[0]].shape[2]
    T = all_protos[subjects[0]].shape[2]
    time_ms = np.arange(T) / sfreq * 1000
    patch_ms = np.arange(N_patches) * (8 / sfreq) * 1000
    boundaries = sorted(set(
        [s for s, e in windows] + [e for s, e in windows]
    ))

    # ── Fig 1: Prototype waveforms ──
    fig, axes = plt.subplots(K, C, figsize=(4 * C, 3 * K), sharex=True)
    fig.suptitle(
        f'Difference-Wave Prototypes ({n_subj} LOSO folds) \u2014 {dataset_label}',
        fontsize=14, fontweight='bold',
    )
    for col, ch in enumerate(channels):
        axes[0, col].set_title(ch, fontsize=13, fontweight='bold')
    # Scale per-fold alpha so overlapping traces/shading stay readable
    trace_alpha = max(0.08, min(0.25, 3.0 / n_subj))
    span_alpha = max(0.005, min(0.04, 0.5 / n_subj))
    for k in range(K):
        s_ms, e_ms = windows[k]
        k_subjects = [s for s in subjects if fold_K[s] > k]
        for c in range(C):
            ax = axes[k, c]
            traces = np.array([all_protos[s][k, c, :] for s in k_subjects])
            for ti, t in enumerate(traces):
                ax.plot(time_ms, t, color=PROTO_COLORS[k], alpha=trace_alpha, lw=0.8)
                subj = k_subjects[ti]
                if subj in per_fold_windows:
                    fw = per_fold_windows[subj]
                    ax.axvspan(fw[k][0], fw[k][1], color=PROTO_COLORS[k],
                               alpha=span_alpha)
            mean = traces.mean(0)
            std = traces.std(0)
            ax.plot(time_ms, mean, color=PROTO_COLORS[k], lw=2.5)
            ax.fill_between(time_ms, mean - std, mean + std,
                            color=PROTO_COLORS[k], alpha=0.15)
            ax.axvspan(s_ms, e_ms, color='gray', alpha=0.15)
            ax.axhline(0, color='gray', lw=0.5, ls='--')
            n_k = len(k_subjects)
            count_note = f' [n={n_k}]' if n_k < n_subj else ''
            if c == 0:
                ax.set_ylabel(
                    f'{PROTO_NAMES[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms){count_note}\n(z-score)',
                    fontsize=10,
                )
    for c in range(C):
        axes[-1, c].set_xlabel('Time (ms)', fontsize=11)
    fig.tight_layout()
    fig.savefig(results_dir / 'fig_prototypes.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved fig_prototypes.png')

    # ── Fig 2: Attention entropy vs AUROC ──
    fig, ax = plt.subplots(figsize=(7, 5.5))
    entropies, auroc_vals = [], []
    for subj in subjects:
        attn = all_attn[subj].mean(axis=1)
        flat = attn.reshape(attn.shape[0], -1)
        norm = flat / (flat.sum(1, keepdims=True) + 1e-10)
        ent = -np.sum(norm * np.log(norm + 1e-10), axis=1)
        entropies.append((ent / np.log(norm.shape[1])).mean())
        auroc_vals.append(aurocs[subj])
    entropies = np.array(entropies)
    auroc_vals = np.array(auroc_vals)
    r, _ = stats.pearsonr(entropies, auroc_vals)

    ax.scatter(entropies, auroc_vals, s=120, c='#c0392b', zorder=5,
               edgecolors='white', linewidth=1.5)
    for i, subj in enumerate(subjects):
        ax.annotate(subj, (entropies[i], auroc_vals[i]),
                    textcoords='offset points', xytext=(8, 8), fontsize=10)
    sl, ic = np.polyfit(entropies, auroc_vals, 1)
    xr = np.linspace(entropies.min() - 0.02, entropies.max() + 0.02, 100)
    ax.plot(xr, sl * xr + ic, '--', color='gray', lw=1.5)
    ax.set_xlabel('Mean Normalized Attention Entropy', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(f'Attention Entropy vs AUROC (r={r:.2f})', fontsize=14,
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(results_dir / 'fig_entropy_vs_auroc.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved fig_entropy_vs_auroc.png')

    # ── Fig 3: Per-prototype attention time course ──
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4.5))
    fig.suptitle(dataset_label, fontsize=14, fontweight='bold')
    for k in range(K):
        ax = axes[k]
        s_ms, e_ms = windows[k]
        k_subjects = [s for s in subjects if fold_K[s] > k]
        n_k = len(k_subjects)
        err_tr, cor_tr = [], []
        for subj in k_subjects:
            attn = all_attn[subj].mean(axis=1)
            lab = all_labels[subj]
            err_tr.append(attn[lab == 1, :, k].mean(0) if (lab == 1).sum() else np.zeros(N_patches))
            cor_tr.append(attn[lab == 0, :, k].mean(0) if (lab == 0).sum() else np.zeros(N_patches))
        err_tr, cor_tr = np.array(err_tr), np.array(cor_tr)
        em, cm = err_tr.mean(0), cor_tr.mean(0)
        es, cs = err_tr.std(0) / np.sqrt(n_k), cor_tr.std(0) / np.sqrt(n_k)
        ax.plot(patch_ms, em, 'r-', lw=2, label='Error trials')
        ax.fill_between(patch_ms, em - es, em + es, color='red', alpha=0.2)
        ax.plot(patch_ms, cm, 'b-', lw=2, label='Correct trials')
        ax.fill_between(patch_ms, cm - cs, cm + cs, color='blue', alpha=0.2)
        ax.axvspan(s_ms, e_ms, color='gray', alpha=0.15)
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_title(f'{PROTO_NAMES[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms)', fontsize=12)
        if k == 0:
            ax.set_ylabel('Attention weight', fontsize=11)
            ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(results_dir / 'fig_attn_timecourse.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved fig_attn_timecourse.png')

    # ── Fig 4: Attention difference overlay ──
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k in range(K):
        k_subjects = [s for s in subjects if fold_K[s] > k]
        n_k = len(k_subjects)
        diffs = []
        for subj in k_subjects:
            attn = all_attn[subj].mean(axis=1)
            lab = all_labels[subj]
            em = attn[lab == 1, :, k].mean(0) if (lab == 1).sum() else np.zeros(N_patches)
            cm = attn[lab == 0, :, k].mean(0) if (lab == 0).sum() else np.zeros(N_patches)
            diffs.append(em - cm)
        diffs = np.array(diffs)
        dm = diffs.mean(0)
        ds = diffs.std(0) / np.sqrt(n_k)
        s_ms, e_ms = windows[k]
        ax.plot(patch_ms, dm, color=PROTO_COLORS[k], lw=2.5,
                label=f'{PROTO_NAMES[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms)')
        ax.fill_between(patch_ms, dm - ds, dm + ds,
                        color=PROTO_COLORS[k], alpha=0.15)
        pi = np.argmax(np.abs(dm))
        pv = dm[pi]
        if abs(pv) > 0.01:
            sign = '+' if pv > 0 else ''
            ax.annotate(f'{sign}{pv:.3f}', (patch_ms[pi], pv),
                        textcoords='offset points',
                        xytext=(5, 8 if pv > 0 else -15),
                        fontsize=9, color=PROTO_COLORS[k])
    for b in boundaries:
        ax.axvline(b, color='gray', lw=0.8, ls=':')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Attention diff (error \u2212 correct)', fontsize=12)
    ax.set_title(dataset_label, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout()
    fig.savefig(results_dir / 'fig_attn_diff_overlay.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved fig_attn_diff_overlay.png')

    # ── Fig 5: Per-subject routing difference ──
    fig, axes = plt.subplots(n_subj, 1, figsize=(10, 2.5 * n_subj),
                             sharex=True)
    if n_subj == 1:
        axes = [axes]
    fig.suptitle(f'Per-Subject Routing Difference \u2014 {dataset_label}',
                 fontsize=14, fontweight='bold', y=1.02)
    for i, subj in enumerate(subjects):
        ax = axes[i]
        attn = all_attn[subj].mean(axis=1)
        lab = all_labels[subj]
        auroc = aurocs[subj]
        subj_K = fold_K[subj]
        for k in range(K):
            if k < subj_K:
                em = attn[lab == 1, :, k].mean(0) if (lab == 1).sum() else np.zeros(N_patches)
                cm = attn[lab == 0, :, k].mean(0) if (lab == 0).sum() else np.zeros(N_patches)
                vals = em - cm
            else:
                vals = np.full(N_patches, np.nan)
            kw = {}
            if i == 0:
                s_ms, e_ms = windows[k]
                kw['label'] = f'{PROTO_NAMES[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms)'
            ax.plot(patch_ms, vals, color=PROTO_COLORS[k], lw=1.8, **kw)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        for b in boundaries:
            ax.axvline(b, color='gray', lw=0.5, ls=':')
        ax.set_ylabel(subj, fontsize=11, fontweight='bold',
                      rotation=0, labelpad=50, va='center')
        ax.text(0.98, 0.85, f'AUROC={auroc:.3f}', transform=ax.transAxes,
                fontsize=10, ha='right', va='top')
    axes[-1].set_xlabel('Time (ms)', fontsize=12)
    handles = [
        plt.Line2D([0], [0], color=PROTO_COLORS[k], lw=2,
                   label=f'{PROTO_NAMES[k]} ({windows[k][0]:.0f}\u2013{windows[k][1]:.0f} ms)')
        for k in range(K)
    ]
    fig.legend(handles=handles, loc='upper center', ncol=K, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(results_dir / 'fig_per_subject_routing.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved fig_per_subject_routing.png')


# =====================================================================
# TP vs TN routing figures (per subject)
# =====================================================================

def _plot_tp_tn(results_dir, dataset_key, channel_config, subject,
                dataset_label, tp_trial, tn_trial, probs, labels, attn,
                proto_raw, proto_windows, sfreq, X_raw, cz_idx,
                title_suffix, output_suffix, detect_ch_name='Cz',
                pos_label='Error', neg_label='Correct'):
    """Shared plotting logic for TP vs TN routing figures."""
    K = proto_raw.shape[0]
    T = proto_raw.shape[2]
    N_patches = attn.shape[2]
    patch_width = T // N_patches
    auroc = float(np.load(results_dir / f'predictions_{subject}.npz')['auroc'])

    time_ms = np.arange(T) / sfreq * 1000
    patch_centers_ms = (np.arange(N_patches) * patch_width + patch_width / 2) / sfreq * 1000

    tp_attn = attn[tp_trial].mean(axis=0)
    tn_attn = attn[tn_trial].mean(axis=0)
    tp_cz = X_raw[tp_trial, cz_idx, :] * 1e6
    tn_cz = X_raw[tn_trial, cz_idx, :] * 1e6
    proto_cz = proto_raw[:, cz_idx, :]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[0.7, 1.2, 1],
                           hspace=0.35, wspace=0.25)

    # Compute per-fold proto names from actual polarity on detection channel
    fold_names = list(PROTO_NAMES)  # default
    if '_auto' in str(results_dir):
        pos_count, neg_count = 0, 0
        fold_names = []
        for k in range(K):
            s_ms, e_ms = proto_windows[k]
            s_samp = int(round(s_ms / 1000 * sfreq))
            e_samp = int(round(e_ms / 1000 * sfreq))
            seg = proto_cz[k, s_samp:e_samp]
            if len(seg) > 0:
                peak_val = seg[int(np.argmax(np.abs(seg)))]
            else:
                peak_val = 0
            if peak_val >= 0:
                pos_count += 1
                fold_names.append(f'P{pos_count}')
            else:
                neg_count += 1
                fold_names.append(f'N{neg_count}')

    # Row 0: Prototype reference
    ax_proto = fig.add_subplot(gs[0, :])
    for k in range(K):
        s_ms, e_ms = proto_windows[k]
        ax_proto.axvspan(s_ms, e_ms, color=PROTO_COLORS[k], alpha=0.20, zorder=1)
    for k in range(K):
        s_ms, e_ms = proto_windows[k]
        s_samp = int(round(s_ms / 1000 * sfreq))
        e_samp = int(round(e_ms / 1000 * sfreq))
        ax_proto.plot(time_ms, proto_cz[k], color=PROTO_COLORS[k],
                      lw=0.6, alpha=0.3, zorder=2)
        ax_proto.plot(time_ms[s_samp:e_samp], proto_cz[k, s_samp:e_samp],
                      color=PROTO_COLORS[k], lw=2.5, zorder=3,
                      label=f'{fold_names[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms)')
    # Mark peak within each prototype's window
    for k in range(K):
        s_ms, e_ms = proto_windows[k]
        s_samp = int(round(s_ms / 1000 * sfreq))
        e_samp = int(round(e_ms / 1000 * sfreq))
        window_signal = proto_cz[k, s_samp:e_samp]
        if len(window_signal) > 0:
            peak_offset = int(np.argmax(np.abs(window_signal)))
            peak_samp = s_samp + peak_offset
            peak_ms = time_ms[peak_samp]
            peak_val = proto_cz[k, peak_samp]
            ax_proto.plot(peak_ms, peak_val, 'o',
                          color=PROTO_COLORS[k], markersize=7, zorder=5,
                          markeredgecolor='white', markeredgewidth=1.0)
            ax_proto.annotate(fold_names[k],
                              (peak_ms, peak_val),
                              textcoords="offset points",
                              xytext=(0, 10 if peak_val >= 0 else -14),
                              ha='center', fontsize=8, fontweight='bold',
                              color=PROTO_COLORS[k])

    ax_proto.axhline(0, color='gray', lw=0.5, ls='--')
    ax_proto.set_xlim(0, time_ms[-1])
    ax_proto.set_ylabel(f'Prototype {detect_ch_name} (z-score)', fontsize=11)
    ax_proto.set_title(f'Diff-Wave Prototypes ({detect_ch_name} channel)', fontsize=12,
                       fontweight='bold')
    ax_proto.legend(fontsize=9, loc='upper right', ncol=K)

    # Trial rows
    trials = [
        (tp_cz, tp_attn, probs[tp_trial], f'{pos_label} trial (TP)', 0),
        (tn_cz, tn_attn, probs[tn_trial], f'{neg_label} trial (TN)', 1),
    ]

    ymax_cz = max(np.abs(tp_cz).max(), np.abs(tn_cz).max()) * 1.15
    ymax_attn = max(tp_attn.max(), tn_attn.max()) * 1.1

    for cz, trial_attn, prob, title, col in trials:
        ax1 = fig.add_subplot(gs[1, col])
        for p in range(N_patches):
            s_samp = p * patch_width
            e_samp = (p + 1) * patch_width
            s_t = time_ms[s_samp]
            e_t = time_ms[min(e_samp, T - 1)]
            for k in range(K):
                w = trial_attn[p, k]
                if w > 0.02:
                    ax1.axvspan(s_t, e_t, color=PROTO_COLORS[k],
                                alpha=w * 0.35, zorder=1, lw=0)
        for p in range(N_patches):
            s_samp = p * patch_width
            e_samp = min((p + 1) * patch_width + 1, T)
            dominant_k = np.argmax(trial_attn[p, :])
            ax1.plot(time_ms[s_samp:e_samp], cz[s_samp:e_samp],
                     color=PROTO_COLORS[dominant_k], lw=2.5, zorder=3,
                     solid_capstyle='round')
        ax1.axhline(0, color='gray', lw=0.5, ls='--', zorder=2)
        ax1.set_title(f'{title}\np({pos_label.lower()}) = {prob:.3f}', fontsize=12,
                      fontweight='bold')
        ax1.set_ylabel(f'{detect_ch_name} amplitude (\u00b5V)', fontsize=11)
        ax1.set_xlim(0, time_ms[-1])
        ax1.set_ylim(-ymax_cz, ymax_cz)

        ax2 = fig.add_subplot(gs[2, col])
        for k in range(K):
            ax2.plot(patch_centers_ms, trial_attn[:, k],
                     color=PROTO_COLORS[k], lw=2.0,
                     label=fold_names[k] if col == 0 else None,
                     zorder=3)
        ax2.set_xlabel('Time (ms)', fontsize=11)
        ax2.set_ylabel('Attention weight', fontsize=11)
        ax2.set_ylim(0, ymax_attn)
        ax2.set_xlim(0, time_ms[-1])
        if col == 0:
            ax2.legend(fontsize=9, loc='upper right')

    fig.suptitle(
        f'Attention Routing: TP vs TN{title_suffix} \u2014 {dataset_label} '
        f'({subject}, AUROC={auroc:.3f})',
        fontsize=13, fontweight='bold', y=1.01)

    fname = f'fig_tp_tn_routing_{subject}{output_suffix}.png'
    fig.savefig(results_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def generate_tp_tn_figures(results_dir, cfg, channel_config,
                           subject, dataset_label):
    """Generate high-confidence and median-confidence TP vs TN figures.

    Args:
        results_dir: path to model results directory
        cfg: dataset config dict (loaded from dataset_config.json)
        channel_config: channel preset name
        subject: subject ID (e.g. "sub-01")
        dataset_label: display label for figure title
    """
    results_dir = Path(results_dir)

    preds = np.load(results_dir / f'predictions_{subject}.npz')
    attn_data = np.load(results_dir / f'attention_{subject}.npz')
    proto_data = np.load(results_dir / f'prototypes_{subject}.npz')

    probs = preds['probs']
    labels = preds['labels']
    auroc = float(preds['auroc'])
    attn = attn_data['attention_weights']
    proto_raw = proto_data['proto_raw']
    proto_windows = [tuple(w) for w in proto_data['proto_windows_ms']]
    sfreq = float(proto_data['sfreq'])

    X_raw, ch_names = load_subject_epochs(cfg, channel_config, subject)
    detect_ch_name = cfg.get('detection_channel', 'Cz')
    cz_idx = ch_names.index(detect_ch_name) if detect_ch_name in ch_names else 1

    # Class labels for figure titles
    pos_label = cfg.get('label_map', {}).get('pos_key', 'error').replace('_', ' ').title()
    neg_label = cfg.get('label_map', {}).get('neg_key', 'correct').replace('_', ' ').title()

    tp_mask = (labels == 1) & (probs >= 0.5)
    tn_mask = (labels == 0) & (probs < 0.5)

    if tp_mask.sum() == 0 or tn_mask.sum() == 0:
        print(f"  WARNING: {subject} has no TP or TN trials, skipping")
        return

    tp_indices = np.where(tp_mask)[0]
    tn_indices = np.where(tn_mask)[0]

    # High confidence: furthest from decision boundary
    tp_high = tp_indices[np.argmax(probs[tp_indices])]
    tn_high = tn_indices[np.argmin(probs[tn_indices])]
    print(f"  {subject} (AUROC={auroc:.4f})")
    print(f"    High-conf TP: idx={tp_high}, prob={probs[tp_high]:.4f}")
    print(f"    High-conf TN: idx={tn_high}, prob={probs[tn_high]:.4f}")
    _plot_tp_tn(results_dir, cfg, channel_config, subject,
                dataset_label, tp_high, tn_high, probs, labels, attn,
                proto_raw, proto_windows, sfreq, X_raw, cz_idx,
                '', '_highconf', detect_ch_name=detect_ch_name,
                pos_label=pos_label, neg_label=neg_label)

    # Median confidence
    tp_sorted = tp_indices[np.argsort(probs[tp_indices])]
    tn_sorted = tn_indices[np.argsort(probs[tn_indices])]
    tp_med = tp_sorted[len(tp_sorted) // 2]
    tn_med = tn_sorted[len(tn_sorted) // 2]
    print(f"    Median TP: idx={tp_med}, prob={probs[tp_med]:.4f}")
    print(f"    Median TN: idx={tn_med}, prob={probs[tn_med]:.4f}")
    _plot_tp_tn(results_dir, cfg, channel_config, subject,
                dataset_label, tp_med, tn_med, probs, labels, attn,
                proto_raw, proto_windows, sfreq, X_raw, cz_idx,
                ' (median conf.)', '_median', detect_ch_name=detect_ch_name,
                pos_label=pos_label, neg_label=neg_label)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate all ERP-XTTN attention figures")
    parser.add_argument("--dataset", required=True,
                        choices=_discover_datasets())
    parser.add_argument("--channels", required=True)
    parser.add_argument("--model", default="erpxttn_fixed",
                        help="Model results directory name (default: erpxttn_fixed)")
    parser.add_argument("--partial", action="store_true",
                        help="Generate figures from partial results (no results.json needed)")
    args = parser.parse_args()

    cfg = load_dataset_config(args.dataset)

    if args.channels not in cfg["variants"]:
        valid = list(cfg["variants"].keys())
        parser.error(f"Invalid channels '{args.channels}' for dataset "
                     f"'{args.dataset}'. Valid: {valid}")
    variant = cfg["variants"][args.channels]
    results_dir = (DATASETS_DIR / cfg["name"] / "results" / "tmin0ms_tmax800ms"
                   / variant / args.model)

    global PROTO_NAMES, PROTO_COLORS
    PROTO_NAMES, PROTO_COLORS = get_proto_config(cfg, results_dir=results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    channel_names = get_channel_names(cfg, args.channels)
    model_label = args.model.upper().replace("_", " ")
    dataset_label = f'{args.dataset.upper()} \u2014 {model_label}'

    if args.partial:
        print(f'=== Partial mode: skipping attention analysis figures (need results.json) ===')
    else:
        print(f'=== Attention analysis figures ===')
        generate_attention_figures(results_dir, dataset_label, channel_names, 256)

    if args.partial:
        # Discover subjects from prediction files
        import glob
        pred_files = sorted(glob.glob(str(results_dir / 'predictions_sub-*.npz')))
        subjects = [Path(p).stem.replace('predictions_', '') for p in pred_files]
        print(f'  Partial mode: found {len(subjects)} subjects')
    else:
        with open(results_dir / 'results.json') as f:
            results = json.load(f)
        subjects = [r['test_subject'] for r in results['folds']]

    print(f'\n=== TP/TN routing figures ===')
    for subj in subjects:
        attn_path = results_dir / f'attention_{subj}.npz'
        if not attn_path.exists():
            print(f'  {subj}: attention file not found, skipping')
            continue
        generate_tp_tn_figures(results_dir, cfg, args.channels,
                               subj, dataset_label)

    print('\nDone!')


if __name__ == '__main__':
    main()
