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

PROTO_NAMES = ['P1-diff', 'Ne-diff', 'Pe-diff', 'LateN-diff']
PROTO_COLORS = ['#e67e22', '#c0392b', '#2980b9', '#27ae60']

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"

DATASET_CONFIG = {
    "bnci": {
        "name": "bnci_horizon_2020_ErrP",
        "subjects": ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"],
        "variants": {
            "midline2": "noref_midline2_rs256_iir_fwd_bp-1-10",
            "midline3": "noref_midline3_rs256_iir_fwd_bp-1-10",
            "full":     "noref_rs256_iir_fwd_bp-1-10",
        },
    },
    "hri": {
        "name": "hri_cursor",
        "subjects": [f"sub-{i:02d}" for i in [2,3,4,5,6,7,8,9,10,11,13]],
        "variants": {
            "midline2": "noref_midline2_iir_fwd_bp-1-10",
            "midline3": "noref_midline3_iir_fwd_bp-1-10",
            "full":     "noref_iir_fwd_bp-1-10",
        },
    },
}


def get_channel_names(dataset_key: str, channel_config: str) -> list[str]:
    """Read channel names from the first epoch file."""
    cfg = DATASET_CONFIG[dataset_key]
    variant = cfg["variants"][channel_config]
    base = DATASETS_DIR / cfg["name"] / "epoched_fif" / "tmin0ms_tmax800ms" / variant
    first_fif = next((base / cfg["subjects"][0]).rglob("*-epo.fif"))
    return mne.read_epochs(str(first_fif), preload=False, verbose=False).ch_names


def load_subject_epochs(dataset_key: str, channel_config: str,
                        subject: str) -> tuple[np.ndarray, list[str]]:
    """Load raw epoch data for a subject. Returns (X_raw, ch_names)."""
    cfg = DATASET_CONFIG[dataset_key]
    variant = cfg["variants"][channel_config]
    base = DATASETS_DIR / cfg["name"] / "epoched_fif" / "tmin0ms_tmax800ms" / variant

    all_X = []
    subj_dir = base / subject
    ch_names = None
    for fif_path in sorted(subj_dir.rglob("*-epo.fif")):
        epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
        if ch_names is None:
            ch_names = epochs.ch_names
        event_id = epochs.event_id
        error_ids = {v for k, v in event_id.items() if "error" in k.lower()}
        correct_ids = {v for k, v in event_id.items() if "correct" in k.lower()}
        keep_ids = error_ids | correct_ids
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
    if per_fold_windows:
        all_w = np.array([per_fold_windows[s] for s in subjects])
        windows = [tuple(np.round(all_w[:, k, :].mean(0), 1))
                   for k in range(all_w.shape[1])]
        print(f'  Dynamic windows (mean across {n_subj} folds): {windows}')
    else:
        from erpxttn import PROTO_WINDOWS_MS
        windows = PROTO_WINDOWS_MS

    K = len(windows)
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
    for k in range(K):
        s_ms, e_ms = windows[k]
        for c in range(C):
            ax = axes[k, c]
            traces = np.array([all_protos[s][k, c, :] for s in subjects])
            for ti, t in enumerate(traces):
                ax.plot(time_ms, t, color=PROTO_COLORS[k], alpha=0.25, lw=0.8)
                if subjects[ti] in per_fold_windows:
                    fw = per_fold_windows[subjects[ti]]
                    ax.axvspan(fw[k][0], fw[k][1], color=PROTO_COLORS[k],
                               alpha=0.04)
            mean = traces.mean(0)
            std = traces.std(0)
            ax.plot(time_ms, mean, color=PROTO_COLORS[k], lw=2.5)
            ax.fill_between(time_ms, mean - std, mean + std,
                            color=PROTO_COLORS[k], alpha=0.15)
            ax.axvspan(s_ms, e_ms, color='gray', alpha=0.15)
            ax.axhline(0, color='gray', lw=0.5, ls='--')
            if c == 0:
                ax.set_ylabel(
                    f'{PROTO_NAMES[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms)\n(z-score)',
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
        err_tr, cor_tr = [], []
        for subj in subjects:
            attn = all_attn[subj].mean(axis=1)
            lab = all_labels[subj]
            err_tr.append(attn[lab == 1, :, k].mean(0) if (lab == 1).sum() else np.zeros(N_patches))
            cor_tr.append(attn[lab == 0, :, k].mean(0) if (lab == 0).sum() else np.zeros(N_patches))
        err_tr, cor_tr = np.array(err_tr), np.array(cor_tr)
        em, cm = err_tr.mean(0), cor_tr.mean(0)
        es, cs = err_tr.std(0) / np.sqrt(n_subj), cor_tr.std(0) / np.sqrt(n_subj)
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
        diffs = []
        for subj in subjects:
            attn = all_attn[subj].mean(axis=1)
            lab = all_labels[subj]
            em = attn[lab == 1, :, k].mean(0) if (lab == 1).sum() else np.zeros(N_patches)
            cm = attn[lab == 0, :, k].mean(0) if (lab == 0).sum() else np.zeros(N_patches)
            diffs.append(em - cm)
        diffs = np.array(diffs)
        dm = diffs.mean(0)
        ds = diffs.std(0) / np.sqrt(n_subj)
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
        for k in range(K):
            em = attn[lab == 1, :, k].mean(0) if (lab == 1).sum() else np.zeros(N_patches)
            cm = attn[lab == 0, :, k].mean(0) if (lab == 0).sum() else np.zeros(N_patches)
            kw = {}
            if i == 0:
                s_ms, e_ms = windows[k]
                kw['label'] = f'{PROTO_NAMES[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms)'
            ax.plot(patch_ms, em - cm, color=PROTO_COLORS[k], lw=1.8, **kw)
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
                title_suffix, output_suffix):
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
                      label=f'{PROTO_NAMES[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms)')
    ax_proto.axhline(0, color='gray', lw=0.5, ls='--')
    ax_proto.set_xlim(0, time_ms[-1])
    ax_proto.set_ylabel('Prototype Cz (z-score)', fontsize=11)
    ax_proto.set_title('Diff-Wave Prototypes (Cz channel)', fontsize=12,
                       fontweight='bold')
    ax_proto.legend(fontsize=9, loc='upper right', ncol=K)

    # Trial rows
    trials = [
        (tp_cz, tp_attn, probs[tp_trial], 'Error trial (TP)', 0),
        (tn_cz, tn_attn, probs[tn_trial], 'Correct trial (TN)', 1),
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
        ax1.set_title(f'{title}\np(error) = {prob:.3f}', fontsize=12,
                      fontweight='bold')
        ax1.set_ylabel('Cz amplitude (\u00b5V)', fontsize=11)
        ax1.set_xlim(0, time_ms[-1])
        ax1.set_ylim(-ymax_cz, ymax_cz)

        ax2 = fig.add_subplot(gs[2, col])
        for k in range(K):
            ax2.plot(patch_centers_ms, trial_attn[:, k],
                     color=PROTO_COLORS[k], lw=2.0,
                     label=PROTO_NAMES[k] if col == 0 else None,
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


def generate_tp_tn_figures(results_dir, dataset_key, channel_config,
                           subject, dataset_label):
    """Generate high-confidence and median-confidence TP vs TN figures.

    Args:
        results_dir: path to model results directory
        dataset_key: "bnci" or "hri"
        channel_config: "midline2", "midline3", or "full"
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

    X_raw, ch_names = load_subject_epochs(dataset_key, channel_config, subject)
    cz_idx = ch_names.index('Cz') if 'Cz' in ch_names else 1

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
    _plot_tp_tn(results_dir, dataset_key, channel_config, subject,
                dataset_label, tp_high, tn_high, probs, labels, attn,
                proto_raw, proto_windows, sfreq, X_raw, cz_idx,
                '', '_highconf')

    # Median confidence
    tp_sorted = tp_indices[np.argsort(probs[tp_indices])]
    tn_sorted = tn_indices[np.argsort(probs[tn_indices])]
    tp_med = tp_sorted[len(tp_sorted) // 2]
    tn_med = tn_sorted[len(tn_sorted) // 2]
    print(f"    Median TP: idx={tp_med}, prob={probs[tp_med]:.4f}")
    print(f"    Median TN: idx={tn_med}, prob={probs[tn_med]:.4f}")
    _plot_tp_tn(results_dir, dataset_key, channel_config, subject,
                dataset_label, tp_med, tn_med, probs, labels, attn,
                proto_raw, proto_windows, sfreq, X_raw, cz_idx,
                ' (median conf.)', '_median')


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate all ERP-XTTN attention figures")
    parser.add_argument("--dataset", required=True, choices=["bnci", "hri"])
    parser.add_argument("--channels", required=True,
                        choices=["midline2", "midline3", "full"])
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    variant = cfg["variants"][args.channels]
    results_dir = (DATASETS_DIR / cfg["name"] / "results" / "tmin0ms_tmax800ms"
                   / variant / "erpxttn")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    channel_names = get_channel_names(args.dataset, args.channels)
    dataset_label = f'{args.dataset.upper()} \u2014 ERP-XTTN'

    print(f'=== Attention analysis figures ===')
    generate_attention_figures(results_dir, dataset_label, channel_names, 256)

    with open(results_dir / 'results.json') as f:
        results = json.load(f)
    subjects = [r['test_subject'] for r in results['folds']]

    print(f'\n=== TP/TN routing figures ===')
    for subj in subjects:
        attn_path = results_dir / f'attention_{subj}.npz'
        if not attn_path.exists():
            print(f'  {subj}: attention file not found, skipping')
            continue
        generate_tp_tn_figures(results_dir, args.dataset, args.channels,
                               subj, dataset_label)

    print('\nDone!')


if __name__ == '__main__':
    main()
