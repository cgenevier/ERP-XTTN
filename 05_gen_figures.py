"""Generate all attention analysis and routing figures for ERP-XTTN.

Usage:
    python 05_gen_figures.py --dataset bnci_errp_013-2015 --channels midline3
    python 05_gen_figures.py --dataset erpcore_n400 --channels midline3_n400

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
from pathlib import Path

import mne
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

PROTO_COLOR_PALETTE = ['#e67e22', '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#1abc9c']

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})

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
        # Check if this is an auto-mode run by looking at the path (path, not
        # just .name, so it also fires when results_dir is a seed subdir like
        # erpxttn_auto/seed-1 — matches the check at fig_tp_tn use below).
        if '_auto' in str(results_dir):
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

    names = cfg.get("proto_names", [f"Proto-{i}" for i in range(4)])
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
    fig, axes = plt.subplots(K, C, figsize=(4.5 * C, 3.2 * K), sharex=True)
    fig.suptitle(
        f'Difference-Wave Prototypes ({n_subj} LOSO folds) \u2014 {dataset_label}',
        fontsize=18, fontweight='bold',
    )
    for col, ch in enumerate(channels):
        axes[0, col].set_title(ch, fontsize=16, fontweight='bold')
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
            ax.tick_params(axis='both', labelsize=12)
            n_k = len(k_subjects)
            count_note = f' [n={n_k}]' if n_k < n_subj else ''
            if c == 0:
                ax.set_ylabel(
                    f'{PROTO_NAMES[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms){count_note}\n(z-score)',
                    fontsize=14,
                )
    for c in range(C):
        axes[-1, c].set_xlabel('Time (ms)', fontsize=14)
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
                    textcoords='offset points', xytext=(8, 8), fontsize=13)
    sl, ic = np.polyfit(entropies, auroc_vals, 1)
    xr = np.linspace(entropies.min() - 0.02, entropies.max() + 0.02, 100)
    ax.plot(xr, sl * xr + ic, '--', color='gray', lw=1.5)
    ax.set_xlabel('Mean Normalized Attention Entropy', fontsize=15)
    ax.set_ylabel('AUROC', fontsize=15)
    ax.set_title(f'Attention Entropy vs AUROC (r={r:.2f})', fontsize=17,
                 fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
    fig.tight_layout()
    fig.savefig(results_dir / 'fig_entropy_vs_auroc.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved fig_entropy_vs_auroc.png')

    # ── Fig 3: Per-prototype attention time course ──
    fig, axes = plt.subplots(1, K, figsize=(5.5 * K, 5.0))
    fig.suptitle(dataset_label, fontsize=18, fontweight='bold')
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
        ax.set_xlabel('Time (ms)', fontsize=15)
        ax.set_title(f'{PROTO_NAMES[k]} ({s_ms:.0f}\u2013{e_ms:.0f} ms)', fontsize=16,
                     fontweight='bold')
        ax.tick_params(axis='both', labelsize=13)
        if k == 0:
            ax.set_ylabel('Attention weight', fontsize=14)
            ax.legend(fontsize=12)
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
                        fontsize=12, color=PROTO_COLORS[k])
    for b in boundaries:
        ax.axvline(b, color='gray', lw=0.8, ls=':')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Time (ms)', fontsize=15)
    ax.set_ylabel('Attention diff (error \u2212 correct)', fontsize=15)
    ax.set_title(dataset_label, fontsize=17, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=12, loc='best')
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
                 fontsize=18, fontweight='bold', y=1.02)
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
        ax.set_ylabel(subj, fontsize=14, fontweight='bold',
                      rotation=0, labelpad=50, va='center')
        ax.text(0.98, 0.85, f'AUROC={auroc:.3f}', transform=ax.transAxes,
                fontsize=13, ha='right', va='top')
    axes[-1].set_xlabel('Time (ms)', fontsize=15)
    handles = [
        plt.Line2D([0], [0], color=PROTO_COLORS[k], lw=2,
                   label=f'{PROTO_NAMES[k]} ({windows[k][0]:.0f}\u2013{windows[k][1]:.0f} ms)')
        for k in range(K)
    ]
    fig.legend(handles=handles, loc='upper center', ncol=K, fontsize=12,
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
                              ha='center', fontsize=11, fontweight='bold',
                              color=PROTO_COLORS[k])

    ax_proto.axhline(0, color='gray', lw=0.5, ls='--')
    ax_proto.set_xlim(0, time_ms[-1])
    ax_proto.set_ylabel(f'Prototype {detect_ch_name} (z-score)', fontsize=14)
    ax_proto.set_title(f'Diff-Wave Prototypes ({detect_ch_name} channel)', fontsize=15,
                       fontweight='bold')
    ax_proto.legend(fontsize=12, loc='upper right', ncol=K)

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
        ax1.set_title(f'{title}\np({pos_label.lower()}) = {prob:.3f}', fontsize=15,
                      fontweight='bold')
        ax1.set_ylabel(f'{detect_ch_name} amplitude (\u00b5V)', fontsize=14)
        ax1.set_xlim(0, time_ms[-1])
        ax1.set_ylim(-ymax_cz, ymax_cz)

        ax2 = fig.add_subplot(gs[2, col])
        for k in range(K):
            ax2.plot(patch_centers_ms, trial_attn[:, k],
                     color=PROTO_COLORS[k], lw=2.0,
                     label=fold_names[k] if col == 0 else None,
                     zorder=3)
        ax2.set_xlabel('Time (ms)', fontsize=14)
        ax2.set_ylabel('Attention weight', fontsize=14)
        ax2.set_ylim(0, ymax_attn)
        ax2.set_xlim(0, time_ms[-1])
        if col == 0:
            ax2.legend(fontsize=12, loc='upper right')

    fig.suptitle(
        f'Attention Routing: TP vs TN{title_suffix} \u2014 {dataset_label} '
        f'({subject}, AUROC={auroc:.3f})',
        fontsize=17, fontweight='bold', y=1.01)

    fname = f'fig_tp_tn_routing_{subject}{output_suffix}.png'
    fig.savefig(results_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# Original ERP-XTTN component palette (P1 orange, N1 green, P2 blue, N2 purple…).
PROTO_COLORS = ['#d1710a', '#2e7d32', '#1565c0', '#8e24aa', '#c62828', '#00838f']


def _component_names(proto_det, windows_ms, sfreq):
    """Name each prototype P1/N1/… from its window's dominant polarity."""
    names, npos, nneg = [], 0, 0
    for k, (s_ms, e_ms) in enumerate(windows_ms):
        s, e = int(s_ms / 1000 * sfreq), int(e_ms / 1000 * sfreq)
        seg = proto_det[k, s:e]
        v = seg[int(np.argmax(np.abs(seg)))] if len(seg) else 0.0
        if v >= 0:
            npos += 1; names.append(f'P{npos}')
        else:
            nneg += 1; names.append(f'N{nneg}')
    return names


def _select_routing(labels, probs, mode):
    """(left, right) trial indices + captions for a confidence/error mode."""
    err, cor = np.where(labels == 1)[0], np.where(labels == 0)[0]

    def cap(i):
        truth = 'Error' if labels[i] == 1 else 'Correct'
        ok = (labels[i] == 1) == (probs[i] >= 0.5)
        return f"{truth} trial — p(err)={probs[i]:.2f} → {'✓' if ok else '✗ MISCLASSIFIED'}"

    if mode in ('high', 'median', 'low'):
        tp = err[probs[err] >= 0.5]; tn = cor[probs[cor] < 0.5]
        if len(tp) == 0 or len(tn) == 0:
            return None
        tp = tp[np.argsort(probs[tp])]; tn = tn[np.argsort(probs[tn])]
        if mode == 'high':
            L, R = tp[-1], tn[0]
        elif mode == 'low':
            L, R = tp[0], tn[-1]
        else:
            L, R = tp[len(tp) // 2], tn[len(tn) // 2]
    elif mode == 'wrong':
        missed = err[probs[err] < 0.5]; alarms = cor[probs[cor] >= 0.5]
        L = missed[np.argmin(probs[missed])] if len(missed) else err[np.argmin(probs[err])]
        R = alarms[np.argmax(probs[alarms])] if len(alarms) else cor[np.argmax(probs[cor])]
    else:  # confident
        L = err[np.argmax(probs[err])]; R = cor[np.argmin(probs[cor])]
    return (int(L), cap(L)), (int(R), cap(R))


def _as_scalar_str(x):
    arr = np.asarray(x)
    return str(arr.item() if arr.shape == () else arr)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _legacy_fusion_metadata(K):
    return {
        "feature_names": ["routing_logit"] + [f"MF_proto{k}" for k in range(K)],
        "feature_slices": {"routing_logit": [0, 1], "mf": [1, 1 + K]},
        "feature_order": ["routing_logit", "mf"],
        "combiner_features": "legacy_routing_mf",
    }


def _fusion_evidence_for_trials(npz_path, Xn, trial_ids):
    """Return per-trial fusion accounting, or None if artifacts are absent.

    Evidence is LR-logit contribution = feature_value * fitted coefficient. The
    feature vector is recomputed from the held-out subject's own frozen
    checkpoint, matching the zero-calibration Stage-2 combiner.
    """
    rpath = Path(npz_path)
    results_dir = rpath.parent
    subject = rpath.stem.replace("routing_", "")
    ckpt_path = results_dir / f"checkpoint_{subject}.pt"
    tf_path = results_dir / f"two_factor_{subject}.npz"
    if not (ckpt_path.exists() and tf_path.exists()):
        return None

    try:
        import torch
        from erpxttn import _fold_features, fusion_feature_metadata, load_frozen_model

        device = torch.device("cpu")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = load_frozen_model(ckpt, device)
        full_feats = _fold_features(model, Xn[np.asarray(trial_ids)].astype(np.float32),
                                    device, bs=max(1, len(trial_ids)))

        tf = np.load(tf_path, allow_pickle=True)
        coef = np.asarray(tf["coef"], dtype=float).reshape(-1)
        intercept = float(tf["intercept"])

        meta = None
        if "feature_metadata_json" in tf.files:
            try:
                meta = json.loads(_as_scalar_str(tf["feature_metadata_json"]))
            except Exception:
                meta = None
        if meta is None:
            meta = fusion_feature_metadata(model)

        if full_feats.shape[1] == coef.shape[0]:
            feats = full_feats
            feature_names = list(meta.get("feature_names", []))
            slices = dict(meta.get("feature_slices", {}))
            combiner_features = _as_scalar_str(
                tf["combiner_features"]) if "combiner_features" in tf.files else \
                "routing_channel_contrast_v2"
        else:
            xb = torch.from_numpy(
                Xn[np.asarray(trial_ids)].astype(np.float32)).to(device)
            legacy_mf = model.compute_mf(xb).cpu().numpy()
            legacy = np.column_stack([full_feats[:, 0], legacy_mf])
            if legacy.shape[1] != coef.shape[0]:
                return None
            feats = legacy
            meta = _legacy_fusion_metadata(model.K)
            feature_names = meta["feature_names"]
            slices = meta["feature_slices"]
            combiner_features = "legacy_routing_mf"

        if not feature_names or len(feature_names) != feats.shape[1]:
            feature_names = [f"f{i}" for i in range(feats.shape[1])]

        probs = tf["probs"] if "probs" in tf.files and len(tf["probs"]) >= max(trial_ids) + 1 else None
        out = {}
        for row, tr in enumerate(trial_ids):
            contrib = feats[row] * coef
            groups = {"intercept": intercept}
            for key in ("routing_logit", "mf", "mf_channel", "mf_contrast"):
                if key in slices:
                    s, e = [int(v) for v in slices[key]]
                    groups[key] = float(contrib[s:e].sum())
            logit = float(contrib.sum() + intercept)
            spatial_idx = []
            for key in ("mf_channel", "mf_contrast"):
                if key in slices:
                    s, e = [int(v) for v in slices[key]]
                    spatial_idx.extend(range(s, e))
            top = sorted(
                [(feature_names[i], float(contrib[i]), float(feats[row, i]))
                 for i in spatial_idx],
                key=lambda t: abs(t[1]), reverse=True)[:5]
            out[int(tr)] = {
                "groups": groups,
                "top_spatial": top,
                "logit": logit,
                "prob": float(probs[tr]) if probs is not None else float(_sigmoid(logit)),
                "combiner_features": combiner_features,
            }
        return out
    except Exception as e:
        print(f"  {subject}: fusion evidence unavailable for routing figure ({e})")
        return None


def _short_feature_label(name, proto_names):
    label = str(name)
    for k, pname in enumerate(proto_names):
        label = label.replace(f"proto{k}", pname)
    label = label.replace("MF_", "")
    label = label.replace("_channel_", ":")
    label = label.replace("_contrast_", ":")
    return label


def _plot_fusion_evidence(ax, evidence, proto_names, xlim=None):
    """Compact decision ledger + top spatial amplitude terms for one trial."""
    group_specs = [
        ("intercept", "bias"),
        ("routing_logit", "routing"),
        ("mf", "scalar MF (legacy)"),
        ("mf_channel", "channel MF"),
        ("mf_contrast", "contrast MF"),
    ]
    labels, vals = [], []
    for key, label in group_specs:
        if key in evidence["groups"]:
            labels.append(label)
            vals.append(float(evidence["groups"][key]))
    y = np.arange(len(vals))
    colors = ["#2e7d32" if v >= 0 else "#b23b3b" for v in vals]
    lim = xlim if xlim is not None else \
        max(0.25, max(abs(v) for v in vals) * 1.25 if vals else 1.0)
    ax.barh(y, vals, color=colors, alpha=0.78, height=0.55)
    ax.axvline(0, color="0.45", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(-lim, lim)
    ax.invert_yaxis()
    ax.set_xlabel("LR logit contribution", fontsize=10)
    ax.set_title(f"fusion p(err)={evidence['prob']:.2f}", fontsize=10.5,
                 fontweight="bold")
    ax.grid(axis="x", color="0.88", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    top = evidence.get("top_spatial", [])
    if top:
        lines = []
        for name, c, _val in top[:4]:
            sign = "+" if c >= 0 else ""
            lines.append(f"{_short_feature_label(name, proto_names)} {sign}{c:.2f}")
        ax.text(1.02, 0.98, "top spatial\n" + "\n".join(lines),
                transform=ax.transAxes, ha="left", va="top", fontsize=8.0,
                color="#555")


def plot_peak_routing(npz_path, out_png, dataset_label='ERP',
                      subject='sub-01', mode='high'):
    """Per-peak routing figure (TP vs TN) from a self-contained routing dump.

    Each detected peak is a unit; similarity m and attention a are per-(peak,
    prototype) stems at the peak's centre. The signal row shades each peak's
    variable-width window coloured by the prototype it most contributes to
    (argmax |a·m|); a filled ▼ marks a resemblance (m>0), a hollow ▽ an
    anti-match (m<0).
    """
    import matplotlib.gridspec as gridspec
    z = np.load(npz_path, allow_pickle=True)
    A, M = z['a'], z['m']
    mask, center, bounds = z['mask'], z['center'], z['bounds']
    X, labels, probs = z['X'], z['labels'], z['probs']
    proto_raw, windows = z['proto_raw'], z['proto_windows_ms']
    sfreq = float(z['sfreq'])
    chans = [str(c) for c in z['channel_names']] if len(z['channel_names']) else ['ch0']
    det_name = str(z['detection_channel']) if 'detection_channel' in z.files else None

    Ntr, P, K = M.shape
    C, T = X.shape[1], X.shape[2]
    det = chans.index(det_name) if (det_name in chans) else \
        int(np.argmax(np.abs(proto_raw).sum(axis=(0, 2))))
    det_name = chans[det] if det < len(chans) else f'ch{det}'
    time_ms = np.arange(T) / sfreq * 1000
    proto_det = proto_raw[:, det, :]
    contrib = A * M
    names = _component_names(proto_det, windows, sfreq)

    sel = _select_routing(labels, probs, mode)
    if sel is None:
        print(f"  {subject} [{mode}]: no TP/TN pair, skipping")
        return
    (tp, tp_lab), (tn, tn_lab) = sel
    fusion_evidence = _fusion_evidence_for_trials(npz_path, X, [tp, tn])
    has_fusion_evidence = fusion_evidence is not None
    fusion_xlim = None
    if has_fusion_evidence:
        vals = []
        for tr in (tp, tn):
            ev = fusion_evidence[int(tr)]
            vals.extend(float(ev["groups"].get(k, 0.0))
                        for k in ("intercept", "routing_logit", "mf",
                                  "mf_channel", "mf_contrast")
                        if k in ev["groups"])
        fusion_xlim = max(0.25, max(abs(v) for v in vals) * 1.25 if vals else 1.0)

    if has_fusion_evidence:
        fig = plt.figure(figsize=(15.5, 16.2))
        gs = gridspec.GridSpec(
            5, 2, figure=fig,
            height_ratios=[0.65, 1.0, 0.88, 0.88, 1.0],
            hspace=0.50, wspace=0.38)
    else:
        fig = plt.figure(figsize=(14, 13.5))
        gs = gridspec.GridSpec(
            4, 2, figure=fig,
            height_ratios=[0.65, 1.0, 0.9, 0.9],
            hspace=0.42, wspace=0.22)

    axp = fig.add_subplot(gs[0, :])
    for k in range(K):
        s_ms, e_ms = windows[k]
        col = PROTO_COLORS[k % len(PROTO_COLORS)]
        axp.axvspan(s_ms, e_ms, color=col, alpha=0.16, zorder=1)
        s, e = int(s_ms / 1000 * sfreq), int(e_ms / 1000 * sfreq)
        axp.plot(time_ms[s:e], proto_det[k, s:e], color=col, lw=2.6,
                 label=f'{names[k]} ({s_ms:.0f}–{e_ms:.0f} ms)')
    axp.axhline(0, color='gray', lw=0.5, ls='--')
    axp.set_xlim(0, time_ms[-1]); axp.set_ylabel(f'Prototype {det_name} (z)', fontsize=13)
    axp.set_title('Difference-Wave Prototypes (grounded templates)',
                  fontsize=14, fontweight='bold')
    axp.legend(fontsize=11, loc='upper right', ncol=K)

    def _vmax(arr):
        vals = [np.abs(arr[t][mask[t]]).max() for t in (tp, tn) if mask[t].any()]
        return (max(vals) if vals else 1.0) * 1.15
    ymax = max(np.abs(X[tp, det]).max(), np.abs(X[tn, det]).max()) * 1.15
    mmax, amax = _vmax(M), _vmax(A)

    for col_i, (tr, lab) in enumerate([(tp, tp_lab), (tn, tn_lab)]):
        sig = X[tr, det]
        vj = np.where(mask[tr])[0]
        cen_ms = center[tr][vj] / sfreq * 1000
        cc, mm = contrib[tr], M[tr]

        ax1 = fig.add_subplot(gs[1, col_i])
        for j in vj:
            lo, hi = int(bounds[tr, j, 0]), int(bounds[tr, j, 1])
            kbest = int(np.argmax(np.abs(cc[j])))
            strong = np.abs(cc[j, kbest]) > 1e-3
            resembles = mm[j, kbest] >= 0
            colr = PROTO_COLORS[kbest % len(PROTO_COLORS)] if strong else '#9e9e9e'
            ax1.axvspan(time_ms[max(0, lo)], time_ms[min(hi, T - 1)], color=colr,
                        alpha=0.20 if strong else 0.06, lw=0, zorder=1)
            pc = int(min(center[tr, j], T - 1))
            ax1.plot(time_ms[pc], sig[pc], 'v', ms=11, zorder=6,
                     markerfacecolor=(colr if resembles else 'white'),
                     markeredgecolor=colr, markeredgewidth=1.6)
            if strong:
                txt = names[kbest] if resembles else '≠' + names[kbest]
                ax1.annotate(txt, (time_ms[pc], sig[pc]), textcoords='offset points',
                             xytext=(0, 12), ha='center', fontsize=10,
                             fontweight='bold', color=colr, zorder=7)
        ax1.plot(time_ms, sig, color='#333', lw=2.0, zorder=3, solid_capstyle='round')
        ax1.axhline(0, color='gray', lw=0.5, ls='--')
        ax1.set_title(f'{lab}   ({len(vj)} peaks)', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'{det_name} (normalized)', fontsize=13)
        ax1.set_xlim(0, time_ms[-1]); ax1.set_ylim(-ymax, ymax * 1.25)

        ax2 = fig.add_subplot(gs[2, col_i])
        for k in range(K):
            colk = PROTO_COLORS[k % len(PROTO_COLORS)]
            ax2.vlines(cen_ms, 0, M[tr, vj, k], color=colk, lw=1.2, alpha=0.5)
            ax2.plot(cen_ms, M[tr, vj, k], 'o', color=colk, ms=6,
                     label=names[k] if col_i == 0 else None)
        ax2.axhline(0, color='gray', lw=0.5, ls='--')
        ax2.set_ylabel('Similarity  m  (per peak)', fontsize=12)
        ax2.set_xlim(0, time_ms[-1]); ax2.set_ylim(-mmax, mmax)
        if col_i == 0:
            ax2.legend(fontsize=10, loc='upper right', ncol=K)

        ax3 = fig.add_subplot(gs[3, col_i])
        for k in range(K):
            colk = PROTO_COLORS[k % len(PROTO_COLORS)]
            ax3.vlines(cen_ms, 0, A[tr, vj, k], color=colk, lw=1.2, alpha=0.5)
            ax3.plot(cen_ms, A[tr, vj, k], 'o', color=colk, ms=6)
        ax3.set_xlabel('Time (ms)', fontsize=13)
        ax3.set_ylabel('Attention  a  (per peak)', fontsize=12)
        ax3.set_xlim(0, time_ms[-1]); ax3.set_ylim(0, amax)

        if has_fusion_evidence:
            ax4 = fig.add_subplot(gs[4, col_i])
            _plot_fusion_evidence(ax4, fusion_evidence[int(tr)], names, fusion_xlim)

    _mtitle = {'confident': 'confident cases', 'high': 'HIGH confidence',
               'median': 'MEDIAN confidence', 'low': 'LOW confidence',
               'wrong': 'MISCLASSIFIED cases'}.get(mode, mode)
    fig.suptitle(f'Peak-Unit Routing ({_mtitle}) — {dataset_label} ({subject})',
                 fontsize=16, fontweight='bold', y=1.005)
    subtitle = ('each variable-width peak gets one m & a per prototype.  '
                'filled ▼ = RESEMBLES proto (m>0);  hollow ▽ = ANTI-matches (m<0)')
    if has_fusion_evidence:
        subtitle += ('; bottom row = Stage-2 LR evidence from routing, MF, '
                     'channel MF, and contrast MF')
    fig.text(0.5, 0.982, subtitle, ha='center', fontsize=10.5,
             color='#555', style='italic')
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def generate_tp_tn_figures(results_dir, cfg, channel_config,
                           subject, dataset_label):
    """High- and median-confidence TP-vs-TN peak-routing figures.

    Reads the self-contained routing_<subject>.npz (a, m, mask, center, bounds,
    X, labels, probs, prototype template + windows) and renders the per-peak
    routing layout. Skips gracefully if the routing dump is absent.
    """
    results_dir = Path(results_dir)
    routing_path = results_dir / f'routing_{subject}.npz'
    if not routing_path.exists():
        print(f"  {subject}: no routing_{subject}.npz, skipping TP/TN routing figure")
        return
    for mode, suffix in [('high', '_highconf'), ('median', '_median')]:
        out = results_dir / f'fig_tp_tn_routing_{subject}{suffix}.png'
        try:
            plot_peak_routing(str(routing_path), str(out),
                              dataset_label=dataset_label, subject=subject, mode=mode)
            print(f"  {subject} [{mode}]: {out.name}")
        except Exception as e:
            print(f"  {subject} [{mode}]: routing figure failed: {e}")


# =====================================================================
# ERP morphology figures (TP vs FN / TN vs FP)
# =====================================================================

def _plot_morphology(fig_path, title, ch_names, time_ms,
                     tp_mean, tp_sem, fn_mean, fn_sem,
                     tn_mean, tn_sem, fp_mean, fp_sem,
                     counts=None, proto_windows=None,
                     pos_label='Error', neg_label='Correct',
                     footer=None):
    """Shared 2 x C plotting for TP/FN and TN/FP morphology comparison.

    Each row is one class (error/correct). Each col is one channel.
    Per panel: two class-conditional means with SEM ribbons plus a thin
    diff trace on the same axis (all in µV).
    """
    C = len(ch_names)
    fig, axes = plt.subplots(2, C, figsize=(4.5 * C, 7), sharex=True,
                             squeeze=False)

    color_a_err = '#2ca02c'  # TP (green = correctly detected)
    color_b_err = '#d62728'  # FN (red = missed)
    color_a_cor = '#1f77b4'  # TN (blue = correctly rejected)
    color_b_cor = '#ff7f0e'  # FP (orange = false alarm)

    rows = [
        {
            'a_mean': tp_mean, 'a_sem': tp_sem,
            'a_label': 'TP (hit)', 'a_color': color_a_err,
            'b_mean': fn_mean, 'b_sem': fn_sem,
            'b_label': 'FN (miss)', 'b_color': color_b_err,
            'class_label': pos_label,
        },
        {
            'a_mean': tn_mean, 'a_sem': tn_sem,
            'a_label': 'TN (correct reject)', 'a_color': color_a_cor,
            'b_mean': fp_mean, 'b_sem': fp_sem,
            'b_label': 'FP (false alarm)', 'b_color': color_b_cor,
            'class_label': neg_label,
        },
    ]

    # Shared y-limits per row for visual comparison across channels
    for r in rows:
        stacked = []
        for arr in (r['a_mean'], r['b_mean']):
            if arr is not None:
                stacked.append(np.abs(arr))
        r['ymax'] = float(np.max(stacked)) * 1.2 if stacked else 1.0

    for row_idx, r in enumerate(rows):
        for c_idx, ch in enumerate(ch_names):
            ax = axes[row_idx, c_idx]

            # Prototype windows as faded colored background
            if proto_windows:
                for wk, win in enumerate(proto_windows):
                    s_ms, e_ms = win
                    ax.axvspan(
                        s_ms, e_ms,
                        color=PROTO_COLOR_PALETTE[wk % len(PROTO_COLOR_PALETTE)],
                        alpha=0.08, zorder=1,
                    )

            if r['a_mean'] is not None:
                ax.plot(time_ms, r['a_mean'][c_idx], color=r['a_color'],
                        lw=2.0, label=r['a_label'], zorder=3)
                if r['a_sem'] is not None:
                    ax.fill_between(
                        time_ms,
                        r['a_mean'][c_idx] - r['a_sem'][c_idx],
                        r['a_mean'][c_idx] + r['a_sem'][c_idx],
                        color=r['a_color'], alpha=0.22, zorder=2,
                    )

            if r['b_mean'] is not None:
                ax.plot(time_ms, r['b_mean'][c_idx], color=r['b_color'],
                        lw=2.0, label=r['b_label'], zorder=3)
                if r['b_sem'] is not None:
                    ax.fill_between(
                        time_ms,
                        r['b_mean'][c_idx] - r['b_sem'][c_idx],
                        r['b_mean'][c_idx] + r['b_sem'][c_idx],
                        color=r['b_color'], alpha=0.22, zorder=2,
                    )

            ax.axhline(0, color='gray', lw=0.5, ls='--', zorder=1)
            ax.set_ylim(-r['ymax'], r['ymax'])

            if row_idx == 0:
                ax.set_title(ch, fontsize=15, fontweight='bold')
            if row_idx == 1:
                ax.set_xlabel('Time (ms)', fontsize=14)
            if c_idx == 0:
                ax.set_ylabel(
                    f'{r["class_label"]} class\namplitude (µV)',
                    fontsize=13,
                )
            if c_idx == C - 1:
                ax.legend(fontsize=11, loc='best', framealpha=0.8)

    title_full = title
    if counts is not None:
        title_full += (
            f'\nTP={counts["tp"]}, FN={counts["fn"]}, '
            f'TN={counts["tn"]}, FP={counts["fp"]}'
        )
    fig.suptitle(title_full, fontsize=16, fontweight='bold', y=1.00)

    if footer:
        fig.text(0.5, -0.01, footer, ha='center', fontsize=12,
                 color='gray', style='italic')

    fig.tight_layout()
    fig.savefig(fig_path, dpi=110, bbox_inches='tight',
                pil_kwargs={'optimize': True, 'compress_level': 9})
    plt.close(fig)


def generate_morphology_figures(results_dir, cfg, channel_config,
                                subjects, dataset_label):
    """Per-subject and aggregate ERP morphology figures: TP vs FN, TN vs FP.

    Uses only the held-out test subject's epochs per fold. Class assignment
    comes from `predictions_<subj>.npz` (probs >= 0.5 threshold, matching
    the balanced-accuracy cutoff used in training).

    For ERPXTTN variants, prototype windows from each fold are overlaid as
    faded background spans. For baselines (EEGNet, xDAWN+RG), no prototype
    background is drawn.
    """
    results_dir = Path(results_dir)

    pos_label = cfg.get('label_map', {}).get('pos_key', 'error') \
        .replace('_', ' ').title()
    neg_label = cfg.get('label_map', {}).get('neg_key', 'correct') \
        .replace('_', ' ').title()

    per_subj = {}
    proto_windows_per_fold = {}
    has_protos = False
    ch_names_canonical = None
    sfreq = 256.0

    for subj in subjects:
        pred_path = results_dir / f'predictions_{subj}.npz'
        if not pred_path.exists():
            print(f'  {subj}: no predictions, skipping')
            continue
        pred = np.load(pred_path)
        probs = pred['probs']
        labels = pred['labels']

        proto_path = results_dir / f'prototypes_{subj}.npz'
        if proto_path.exists():
            has_protos = True
            pdata = np.load(proto_path)
            proto_windows_per_fold[subj] = [
                (float(w[0]), float(w[1])) for w in pdata['proto_windows_ms']
            ]
            sfreq = float(pdata['sfreq'])

        X_raw, ch_names = load_subject_epochs(cfg, channel_config, subj)
        X_uV = X_raw * 1e6  # Volts -> microvolts

        # For "full" variants (30+ channels), plotting every channel
        # produces unreadable 20k-px-wide figures. Clamp to the dataset's
        # detection channel only — matches the existing TP/TN routing
        # figure convention for full variants.
        if channel_config == 'full':
            det_name = cfg.get('detection_channel', 'Cz')
            det_idx = ch_names.index(det_name) if det_name in ch_names else min(1, len(ch_names) - 1)
            X_uV = X_uV[:, det_idx:det_idx + 1, :]
            ch_names = [ch_names[det_idx]]

        if ch_names_canonical is None:
            ch_names_canonical = ch_names

        if len(probs) != X_uV.shape[0]:
            print(f'  WARNING: {subj} predictions ({len(probs)}) do not match '
                  f'epochs ({X_uV.shape[0]}), skipping')
            continue

        preds = (probs >= 0.5).astype(int)
        tp_mask = (labels == 1) & (preds == 1)
        fn_mask = (labels == 1) & (preds == 0)
        tn_mask = (labels == 0) & (preds == 0)
        fp_mask = (labels == 0) & (preds == 1)

        def _mean(mask):
            return X_uV[mask].mean(axis=0) if mask.sum() > 0 else None

        def _sem(mask):
            n = int(mask.sum())
            if n <= 1:
                return None
            return X_uV[mask].std(axis=0, ddof=1) / np.sqrt(n)

        per_subj[subj] = {
            'tp_mean': _mean(tp_mask), 'tp_sem': _sem(tp_mask),
            'fn_mean': _mean(fn_mask), 'fn_sem': _sem(fn_mask),
            'tn_mean': _mean(tn_mask), 'tn_sem': _sem(tn_mask),
            'fp_mean': _mean(fp_mask), 'fp_sem': _sem(fp_mask),
            'counts': {
                'tp': int(tp_mask.sum()), 'fn': int(fn_mask.sum()),
                'tn': int(tn_mask.sum()), 'fp': int(fp_mask.sum()),
            },
            'auroc': float(pred['auroc']),
            'n_times': X_uV.shape[2],
        }

    if not per_subj:
        print('  No subjects with predictions; skipping morphology figures.')
        return

    T = per_subj[next(iter(per_subj))]['n_times']
    time_ms = np.arange(T) / sfreq * 1000.0
    C = len(ch_names_canonical)

    # ── Per-subject figures ──
    for subj, s in per_subj.items():
        _plot_morphology(
            fig_path=results_dir / f'fig_morphology_{subj}.png',
            title=(f'Morphology by outcome — {dataset_label} '
                   f'({subj}, AUROC={s["auroc"]:.3f})'),
            ch_names=ch_names_canonical, time_ms=time_ms,
            tp_mean=s['tp_mean'], tp_sem=s['tp_sem'],
            fn_mean=s['fn_mean'], fn_sem=s['fn_sem'],
            tn_mean=s['tn_mean'], tn_sem=s['tn_sem'],
            fp_mean=s['fp_mean'], fp_sem=s['fp_sem'],
            counts=s['counts'],
            proto_windows=proto_windows_per_fold.get(subj),
            pos_label=pos_label, neg_label=neg_label,
        )
        print(f'  Saved fig_morphology_{subj}.png  '
              f'(TP={s["counts"]["tp"]}, FN={s["counts"]["fn"]}, '
              f'TN={s["counts"]["tn"]}, FP={s["counts"]["fp"]})')

    # ── Aggregate figure ──
    def _agg(key):
        means = [s[key] for s in per_subj.values() if s[key] is not None]
        if not means:
            return None, None, 0
        stacked = np.stack(means, axis=0)
        n = stacked.shape[0]
        grand = stacked.mean(axis=0)
        if n <= 1:
            return grand, None, n
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(n)
        return grand, sem, n

    tp_a, tp_a_sem, n_tp = _agg('tp_mean')
    fn_a, fn_a_sem, n_fn = _agg('fn_mean')
    tn_a, tn_a_sem, n_tn = _agg('tn_mean')
    fp_a, fp_a_sem, n_fp = _agg('fp_mean')

    agg_proto_windows = None
    if has_protos and proto_windows_per_fold:
        fold_ws = list(proto_windows_per_fold.values())
        K = max(len(w) for w in fold_ws)
        agg_proto_windows = []
        for k in range(K):
            k_wins = [w[k] for w in fold_ws if len(w) > k]
            if k_wins:
                agg_proto_windows.append((
                    float(np.mean([w[0] for w in k_wins])),
                    float(np.mean([w[1] for w in k_wins])),
                ))

    n_subjects = len(per_subj)
    footer_bits = [f'n_subjects={n_subjects}']
    if min(n_tp, n_fn, n_tn, n_fp) < n_subjects:
        footer_bits.append(
            f'subject contributions TP/FN/TN/FP = '
            f'{n_tp}/{n_fn}/{n_tn}/{n_fp} '
            f'(subjects with 0 trials in a category drop out of that mean)'
        )
    footer = '  |  '.join(footer_bits)

    _plot_morphology(
        fig_path=results_dir / 'fig_morphology_aggregate.png',
        title=(f'Aggregate morphology by outcome — {dataset_label} '
               f'({n_subjects} subjects, grand-average ± SEM across subjects)'),
        ch_names=ch_names_canonical, time_ms=time_ms,
        tp_mean=tp_a, tp_sem=tp_a_sem,
        fn_mean=fn_a, fn_sem=fn_a_sem,
        tn_mean=tn_a, tn_sem=tn_a_sem,
        fp_mean=fp_a, fp_sem=fp_a_sem,
        counts=None, proto_windows=agg_proto_windows,
        pos_label=pos_label, neg_label=neg_label,
        footer=footer,
    )
    print(f'  Saved fig_morphology_aggregate.png  '
          f'(subject contributions TP/FN/TN/FP = {n_tp}/{n_fn}/{n_tn}/{n_fp})')


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate all ERP-XTTN attention figures")
    parser.add_argument("--dataset", required=True,
                        choices=_discover_datasets())
    parser.add_argument("--channels", required=True)
    parser.add_argument("--model", default="erpxttn_peak",
                        help="Model results directory name (default: erpxttn_peak)")
    parser.add_argument("--partial", action="store_true",
                        help="Generate figures from partial results (no results.json needed)")
    parser.add_argument("--morphology-only", action="store_true",
                        help="Only generate morphology figures (skip attention and TP/TN routing)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Reference seed subdir to read for seeded runs "
                             "(default: 1; matches 06/07 REF_SEED)")
    args = parser.parse_args()

    cfg = load_dataset_config(args.dataset)

    if args.channels not in cfg["variants"]:
        valid = list(cfg["variants"].keys())
        parser.error(f"Invalid channels '{args.channels}' for dataset "
                     f"'{args.dataset}'. Valid: {valid}")
    variant = cfg["variants"][args.channels]
    results_dir = (DATASETS_DIR / cfg["name"] / "results" / "tmin0ms_tmax800ms"
                   / variant / args.model)

    # Seed-aware: seeded runs store per-fold results under <model>/seed-N/. If the
    # model dir has no results.json but the requested seed subdir exists, descend
    # into it (matches 06/07 REF_SEED). Flat/legacy layouts already have
    # results.json at the model root and are left unchanged.
    if not (results_dir / "results.json").exists():
        seed_dir = results_dir / f"seed-{args.seed}"
        if seed_dir.exists():
            results_dir = seed_dir

    global PROTO_NAMES, PROTO_COLORS
    PROTO_NAMES, PROTO_COLORS = get_proto_config(cfg, results_dir=results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    channel_names = get_channel_names(cfg, args.channels)
    model_label = args.model.upper().replace("_", " ")
    dataset_label = f'{args.dataset.upper()} \u2014 {model_label}'

    # Discover subjects (results.json takes priority, fallback to prediction files)
    if args.partial or not (results_dir / 'results.json').exists():
        import glob
        pred_files = sorted(glob.glob(str(results_dir / 'predictions_sub-*.npz')))
        subjects = [Path(p).stem.replace('predictions_', '') for p in pred_files]
        print(f'  Discovered {len(subjects)} subjects from prediction files')
    else:
        with open(results_dir / 'results.json') as f:
            results = json.load(f)
        subjects = [r['test_subject'] for r in results['folds']]

    has_routing = any(
        (results_dir / f'routing_{s}.npz').exists() for s in subjects
    )

    if args.morphology_only:
        print(f'=== --morphology-only: skipping TP/TN routing ===')
    elif has_routing:
        print(f'\n=== TP/TN peak-routing figures ===')
        for subj in subjects:
            if not (results_dir / f'routing_{subj}.npz').exists():
                print(f'  {subj}: no routing dump, skipping')
                continue
            generate_tp_tn_figures(results_dir, cfg, args.channels,
                                   subj, dataset_label)
    else:
        print(f'\n=== Skipping TP/TN routing figures '
              f'(no routing files — expected for baselines) ===')

    print(f'\n=== Morphology figures (TP vs FN / TN vs FP) ===')
    generate_morphology_figures(results_dir, cfg, args.channels,
                                subjects, dataset_label)

    print('\nDone!')


if __name__ == '__main__':
    main()
