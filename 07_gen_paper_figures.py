#!/usr/bin/env python3
"""07_gen_paper_figures.py — publication-grade figures for the extension.

Reads from analysis_summary.json + per-subject results files. Produces:

  paper_figures/fig_tax_drivers.png
  paper_figures/fig_tpfp_tptn.png
  paper_figures/fig_morphology_hri_errp_3ch_Cz.png
  paper_figures/fig_morphology_bnci_errp_3ch_Cz.png         (supp replication)
  paper_figures/fig_routing_<dataset>_<channels>.png  (×9, 3-channel only)

  paper_figures/morphology_cache/{hri_errp,bnci_errp}_3ch_Cz.npz
    Per-subject TP/FN/TN/FP grand-mean waveforms at Cz, plus n-counts.
    Computed once from epoched_fif; afterwards, the figure script reads
    only the cache.

Run with no arguments. Re-running re-uses the morphology cache if present.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent
DATASETS_DIR = REPO / 'datasets'
OUT_DIR = REPO / 'paper_figures'
CACHE_DIR = OUT_DIR / 'morphology_cache'
ANALYSIS = REPO / 'analysis_summary.json'

# --------------------------------------------------------------
# Style
# --------------------------------------------------------------
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'savefig.dpi': 300,
    'figure.dpi': 100,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# --------------------------------------------------------------
# Datasets — same ordering as main paper Table 2
# --------------------------------------------------------------
# (display, dataset_dir, 3ch_variant_dir, full_variant_dir, 3ch_combo_key, full_combo_key, detection_channel_idx_in_3ch)
DATASETS = [
    ('ERN',       'erpcore_ern',         'noref_midline3_ern_rs256_iir_fwd_bp-1-10',     'noref_rs256_iir_fwd_bp-1-10', 'erpcore_ern|midline3_ern',     'erpcore_ern|full',         1),  # FCz, Cz, Pz
    ('LRP',       'erpcore_lrp',         'noref_lateral3_lrp_rs256_iir_fwd_bp-1-10',     'noref_rs256_iir_fwd_bp-1-10', 'erpcore_lrp|lateral3_lrp',     'erpcore_lrp|full',         0),  # C3, Cz, C4 — det=C3
    ('HRI ErrP',  'hri_errp_cursor',     'noref_midline3_iir_fwd_bp-1-10',               'noref_iir_fwd_bp-1-10',       'hri_errp_cursor|midline3',     'hri_errp_cursor|full',     1),
    ('BNCI ErrP', 'bnci_errp_013-2015',  'noref_midline3_rs256_iir_fwd_bp-1-10',         'noref_rs256_iir_fwd_bp-1-10', 'bnci_errp_013-2015|midline3',  'bnci_errp_013-2015|full',  1),
    ('N170',      'erpcore_n170',        'noref_occipital3_n170_rs256_iir_fwd_bp-1-10',  'noref_rs256_iir_fwd_bp-1-10', 'erpcore_n170|occipital3_n170', 'erpcore_n170|full',        0),  # P7, Oz, P8 — det=P7
    ('P300',      'erpcore_p300',        'noref_midline3_rs256_iir_fwd_bp-1-10',         'noref_rs256_iir_fwd_bp-1-10', 'erpcore_p300|midline3',        'erpcore_p300|full',        1),
    ('N2pc',      'erpcore_n2pc',        'noref_posterior3_n2pc_rs256_iir_fwd_bp-1-10',  'noref_rs256_iir_fwd_bp-1-10', 'erpcore_n2pc|posterior3_n2pc', 'erpcore_n2pc|full',        0),  # PO7, Pz, PO8 — det=PO7
    ('MMN',       'erpcore_mmn',         'noref_midline3_rs256_iir_fwd_bp-1-10',         'noref_rs256_iir_fwd_bp-1-10', 'erpcore_mmn|midline3',         'erpcore_mmn|full',         1),
    ('N400',      'erpcore_n400',        'noref_midline3_n400_rs256_iir_fwd_bp-1-10',    'noref_rs256_iir_fwd_bp-1-10', 'erpcore_n400|midline3_n400',   'erpcore_n400|full',        0),  # Cz, CPz, Pz — det=Cz idx 0
]


def load_analysis():
    return json.load(open(ANALYSIS))


# ====================================================================
# A. Tax-driver scatter (2×4)
# ====================================================================

def fig_tax_drivers():
    summary = load_analysis()
    combos = summary['combos']

    rows_full, rows_3ch = [], []
    for name, ds_dir, c3, cf, k3, kf, _ in DATASETS:
        cf_obj = combos[kf]
        c3_obj = combos[k3]
        for src, rows in [(cf_obj, rows_full), (c3_obj, rows_3ch)]:
            rows.append({
                'name': name,
                'tax_eeg': src['tax']['eegnet']['mean_delta'],
                'tax_xdr': src['tax']['xdawn_rg']['mean_delta'],
                'H_attn':  src['attention_entropy']['mean_entropy'],
                'R_disc':  src['routing_discriminability']['mean_distance'],
                'P_stab':  src['proto_stability']['mean_pairwise_r'],
                'SNR':     src['snr_proxy']['mean_snr'],
            })

    # Color from the established prototype palette (defined later in file)
    palette = ['#e67e22', '#c0392b', '#2980b9', '#27ae60']
    PREDICTORS = [
        ('H_attn', 'Attention\nentropy',         palette[0]),
        ('R_disc', 'Routing\ndiscriminability',  palette[1]),
        ('P_stab', 'Prototype\nstability',       palette[2]),
        ('SNR',    'SNR\nproxy',                 palette[3]),
    ]
    TAXES = [
        ('tax_eeg', r'$\Delta$ vs EEGNet'),
        ('tax_xdr', r'$\Delta$ vs xDAWN+RG'),
    ]

    # Compute ρ values: rho[tax_key][predictor_key] = (rho_full, rho_3ch)
    rho = {tk: {} for tk, _ in TAXES}
    for tax_key, _ in TAXES:
        for pkey, _, _ in PREDICTORS:
            x_full = np.array([r[pkey] for r in rows_full])
            y_full = np.array([r[tax_key] for r in rows_full])
            x_3ch  = np.array([r[pkey] for r in rows_3ch])
            y_3ch  = np.array([r[tax_key] for r in rows_3ch])
            rho_full, _ = spearmanr(x_full, y_full)
            rho_3ch,  _ = spearmanr(x_3ch,  y_3ch)
            rho[tax_key][pkey] = (rho_full, rho_3ch)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5), sharey=True)

    n_pred = len(PREDICTORS)
    x_pos = np.arange(n_pred)
    YLIM = (-0.55, 0.85)

    for ax_i, (tax_key, tax_label) in enumerate(TAXES):
        ax = axes[ax_i]

        ax.axhline(0, color='gray', lw=0.7, ls='--', zorder=1)

        for xi, (pkey, _, pcolor) in enumerate(PREDICTORS):
            rho_full, rho_3ch = rho[tax_key][pkey]
            xp = x_pos[xi]

            # Filled = Full montage, open = 3-channel
            ax.scatter(xp, rho_full, marker='o', s=70,
                       facecolor=pcolor, edgecolor=pcolor,
                       linewidth=1.3, zorder=3)
            ax.scatter(xp, rho_3ch, marker='o', s=70,
                       facecolor='white', edgecolor=pcolor,
                       linewidth=1.3, zorder=3)

            # Place full label to the right, 3ch label to the left of the
            # dot — guarantees no overlap even when paired values are close.
            ax.annotate(f'{rho_full:+.2f}', (xp, rho_full),
                        textcoords='offset points', xytext=(9, 0),
                        fontsize=7.5, ha='left', va='center', color='black')
            ax.annotate(f'{rho_3ch:+.2f}', (xp, rho_3ch),
                        textcoords='offset points', xytext=(-9, 0),
                        fontsize=7.5, ha='right', va='center', color='black')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([p[1] for p in PREDICTORS], fontsize=8.5)
        ax.set_xlim(-0.55, n_pred - 0.45)
        ax.set_ylim(*YLIM)
        ax.set_title(tax_label, fontsize=10.5)
        if ax_i == 0:
            ax.set_ylabel(r'Spearman $\rho$', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', length=0)

    # 'n = 9 per subset' — bottom-right corner of right panel, unobtrusive
    axes[1].text(0.98, 0.02, 'n = 9 per subset',
                 transform=axes[1].transAxes,
                 fontsize=7.5, ha='right', va='bottom',
                 color='gray')

    # Legend: one column per predictor (Full filled on top, 3-channel open
    # below). Handles are interleaved [Full, 3ch] per predictor because
    # matplotlib fills the legend column-major.
    from matplotlib.lines import Line2D
    legend_handles = []
    for _, plabel, pcolor in PREDICTORS:
        plabel_flat = plabel.replace('\n', ' ')
        legend_handles.append(
            Line2D([0], [0], marker='o', color='none',
                   markerfacecolor=pcolor, markeredgecolor=pcolor,
                   markersize=7, lw=0,
                   label=f'{plabel_flat} — Full')
        )
        legend_handles.append(
            Line2D([0], [0], marker='o', color='none',
                   markerfacecolor='white', markeredgecolor=pcolor,
                   markersize=7, lw=0,
                   label=f'{plabel_flat} — 3-channel')
        )
    fig.legend(handles=legend_handles,
               loc='upper center', frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), ncol=4,
               handletextpad=0.4, columnspacing=1.6)

    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out = OUT_DIR / 'fig_tax_drivers.png'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  wrote {out}')


# ====================================================================
# B. TP↔FP / TP↔TN bar chart
# ====================================================================

def fig_tpfp_tptn():
    summary = load_analysis()
    combos = summary['combos']

    # Order: 18 columns, grouped as Dataset / Channel
    bars = []  # list of (dataset_label, channel_label, fp_mean, fp_lo, fp_hi, tn_mean, tn_lo, tn_hi)
    for name, ds_dir, c3, cf, k3, kf, _ in DATASETS:
        for ch_label, key in [('Full', kf), ('3ch', k3)]:
            c = combos[key]['tp_fp_tn_corr']
            fp = c['tp_fp']
            tn = c['tp_tn']
            bars.append((name, ch_label,
                         fp['mean_r'], fp['ci_low'], fp['ci_high'],
                         tn['mean_r'], tn['ci_low'], tn['ci_high']))

    fig, ax = plt.subplots(figsize=(13.5, 5.2))

    x = np.arange(len(bars))
    width = 0.38

    fp_means = [b[2] for b in bars]
    fp_err_low  = [b[2] - b[3] for b in bars]
    fp_err_high = [b[4] - b[2] for b in bars]
    tn_means = [b[5] for b in bars]
    tn_err_low  = [b[5] - b[6] for b in bars]
    tn_err_high = [b[7] - b[5] for b in bars]

    ax.bar(x - width/2, fp_means, width,
           yerr=[fp_err_low, fp_err_high],
           capsize=2.5, color='#404040', edgecolor='black',
           linewidth=0.6, label='TP↔FP r', error_kw={'lw': 0.6})
    ax.bar(x + width/2, tn_means, width,
           yerr=[tn_err_low, tn_err_high],
           capsize=2.5, color='white', edgecolor='black',
           linewidth=0.6, hatch='////', label='TP↔TN r',
           error_kw={'lw': 0.6})

    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('Pearson r (cross-subject mean, 95% bootstrap CI)', fontsize=9.5)
    ax.set_xticks(x)
    # Per-bar tick labels: just the channel condition
    ch_labels = [b[1] for b in bars]
    ax.set_xticklabels(ch_labels, fontsize=7.5, rotation=0)
    ax.tick_params(axis='x', length=2, pad=2)

    # Dataset names below, centered between each pair, in a second tier
    for i in range(0, len(bars), 2):
        ax.text(i + 0.5, -0.85, bars[i][0],
                ha='center', va='top', fontsize=9, transform=ax.transData)

    # Dividers between datasets
    for i in range(2, len(bars), 2):
        ax.axvline(i - 0.5, color='gray', lw=0.5, ls=':', alpha=0.7)

    ax.set_xlim(-0.7, len(bars) - 0.3)
    ax.set_ylim(-0.7, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=False, ncol=2, fontsize=9)
    ax.set_title('TP↔FP and TP↔TN waveform correlations across datasets',
                 fontsize=11, pad=10)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT_DIR / 'fig_tpfp_tptn.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


# ====================================================================
# C. Routing combined figure (3-panel) per component
# ====================================================================

def _patch_centers_ms(N_patches, sfreq, pw=8):
    return (np.arange(N_patches) + 0.5) * (pw / sfreq) * 1000.0


def _load_routing_data(ds_dir, var_dir):
    """Return dict with: subjects, attn[subj], labels[subj], aurocs[subj],
    fold_K[subj], windows (mean across folds), N_patches, K, sfreq."""
    rdir = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / 'erpxttn_auto'
    res = json.load(open(rdir / 'results.json'))
    subjects = [r['test_subject'] for r in res['folds']]
    aurocs = {r['test_subject']: r['test_auroc'] for r in res['folds']}

    attn, labels = {}, {}
    fold_windows = {}
    sfreq = None
    for s in subjects:
        d = np.load(rdir / f'attention_{s}.npz')
        attn[s] = d['attention_weights']  # (trials, H, N_patches, K)
        labels[s] = d['labels']
        p = np.load(rdir / f'prototypes_{s}.npz')
        fold_windows[s] = [tuple(w) for w in p['proto_windows_ms']]
        sfreq = float(p['sfreq'])

    fold_K = {s: len(fold_windows[s]) for s in subjects}
    K = max(fold_K.values())
    windows = []
    for k in range(K):
        wks = [fold_windows[s][k] for s in subjects if fold_K[s] > k]
        windows.append(tuple(np.round(np.mean(wks, axis=0), 1)))

    N_patches = attn[subjects[0]].shape[2]
    return dict(subjects=subjects, attn=attn, labels=labels, aurocs=aurocs,
                fold_K=fold_K, windows=windows, N_patches=N_patches, K=K,
                sfreq=sfreq, fold_windows=fold_windows)


def _format_class_label(key):
    """Convert label_map key (e.g. 'target_left', 'nontarget') into a display label."""
    SPECIAL = {'nontarget': 'Non-target'}
    if key in SPECIAL:
        return SPECIAL[key]
    s = key.replace('_', ' ')
    return s[:1].upper() + s[1:]


def fig_routing_combined(ds_label, ds_dir, var_dir, out_path):
    """Two-panel routing figure: per-prototype class-averaged timecourses
    on top, per-subject Δ-attention heatmaps on bottom.

    Prototype names come from dataset_config.proto_names (padded with
    Proto-N where K exceeds the named slots). Per-prototype windows in
    panel A are shaded with PROTO_COLOR_PALETTE to match
    fig_prototypes_all_datasets.png. Panel A shares a y-axis across all
    prototype subplots so attention magnitudes are directly comparable.
    Heatmap colorbar label and class legend use paradigm-specific names.
    """
    R = _load_routing_data(ds_dir, var_dir)
    subjects = R['subjects']
    K = R['K']
    N = R['N_patches']
    sfreq = R['sfreq']
    patch_ms = _patch_centers_ms(N, sfreq)

    cfg = json.load(open(DATASETS_DIR / ds_dir / 'dataset_config.json'))
    pos_label = _format_class_label(cfg['label_map']['pos_key'])
    neg_label = _format_class_label(cfg['label_map']['neg_key'])

    # Prototype names from dataset config (pad with Proto-N if K exceeds)
    cfg_proto = list(cfg.get('proto_names', []))
    while len(cfg_proto) < K:
        cfg_proto.append(f'Proto-{len(cfg_proto) + 1}')
    proto_names = cfg_proto[:K]

    # Per-subject mean attention per class per prototype: (subj, N, K)
    em_arr = np.full((len(subjects), N, K), np.nan)
    cm_arr = np.full((len(subjects), N, K), np.nan)
    for i, s in enumerate(subjects):
        a = R['attn'][s].mean(axis=1)
        Ks = R['fold_K'][s]
        lab = R['labels'][s]
        if (lab == 1).sum():
            em_arr[i, :, :Ks] = a[lab == 1].mean(axis=0)
        if (lab == 0).sum():
            cm_arr[i, :, :Ks] = a[lab == 0].mean(axis=0)
    diff_arr = em_arr - cm_arr

    # Sort subjects by AUROC descending for heatmap rows
    auroc_vals = np.array([R['aurocs'][s] for s in subjects])
    order = np.argsort(-auroc_vals)
    diff_sorted = diff_arr[order]
    subj_sorted = [subjects[i] for i in order]

    # Symmetric color limit clipped to 95th percentile of |Δ|
    abs_diff = np.abs(diff_arr[~np.isnan(diff_arr)])
    vlim = float(np.percentile(abs_diff, 95)) if abs_diff.size else 1e-3
    if vlim < 1e-6:
        vlim = 1e-3

    # Layout: top row K timecourses (shared y), bottom row K heatmaps + colorbar
    n_subj = len(subjects)
    heat_h = max(0.10 * n_subj, 1.5)
    fig = plt.figure(figsize=(2.6 * K + 1.5, 3.4 + heat_h + 1.2))
    gs = GridSpec(
        nrows=2, ncols=K + 1,
        height_ratios=[3.0, heat_h],
        width_ratios=[1.0] * K + [0.05],
        hspace=0.18, wspace=0.20,
        top=0.91, bottom=0.06,
    )

    # --- Panel A: per-prototype class-averaged timecourses ---
    axes_A = []
    for k in range(K):
        sharey = axes_A[0] if axes_A else None
        ax = fig.add_subplot(gs[0, k], sharey=sharey)
        axes_A.append(ax)
        proto_color = PROTO_COLOR_PALETTE[k % len(PROTO_COLOR_PALETTE)]
        s_ms, e_ms = R['windows'][k]
        ax.axvspan(s_ms, e_ms, color=proto_color, alpha=0.18, zorder=1)

        em_k = em_arr[:, :, k]
        cm_k = cm_arr[:, :, k]
        n_k = int(np.sum(~np.isnan(em_k[:, 0])))
        if n_k < 1:
            ax.text(0.5, 0.5, 'n/a', transform=ax.transAxes,
                    ha='center', va='center')
        else:
            em_m = np.nanmean(em_k, axis=0)
            em_s = np.nanstd(em_k, axis=0, ddof=1) / np.sqrt(n_k)
            cm_m = np.nanmean(cm_k, axis=0)
            cm_s = np.nanstd(cm_k, axis=0, ddof=1) / np.sqrt(n_k)
            ax.plot(patch_ms, em_m, color='black', lw=1.6, ls='-',
                    label=pos_label, zorder=4)
            ax.fill_between(patch_ms, em_m - em_s, em_m + em_s,
                            color='black', alpha=0.18, lw=0, zorder=2)
            ax.plot(patch_ms, cm_m, color='#888888', lw=1.6, ls='--',
                    label=neg_label, zorder=3)
            ax.fill_between(patch_ms, cm_m - cm_s, cm_m + cm_s,
                            color='#888888', alpha=0.18, lw=0, zorder=2)
        ax.set_title(f'{proto_names[k]}  ({s_ms:.0f}–{e_ms:.0f} ms)',
                     fontsize=9.5, color=proto_color, fontweight='bold')
        ax.set_xlim(patch_ms[0] - 5, patch_ms[-1] + 5)
        if k == 0:
            ax.set_ylabel('Attention weight', fontsize=9)
            ax.legend(fontsize=8, loc='best', framealpha=0.85)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', labelsize=8)

    # --- Panel B: per-subject heatmap (K side-by-side) ---
    cbar_im = None
    for k in range(K):
        ax = fig.add_subplot(gs[1, k])
        data = diff_sorted[:, :, k]
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                       vmin=-vlim, vmax=vlim,
                       extent=[patch_ms[0], patch_ms[-1],
                               len(subjects) - 0.5, -0.5],
                       interpolation='nearest')
        cbar_im = im
        ax.set_xlabel('Time (ms)', fontsize=9)
        if k == 0:
            n = len(subj_sorted)
            ax.set_yticks(np.arange(n))
            # Shrink font for large n so all 40 fit; size scales linearly.
            label_fs = 7 if n <= 12 else (5.5 if n <= 25 else 4.5)
            ax.set_yticklabels(subj_sorted, fontsize=label_fs)
            ax.set_ylabel('Subject (sorted by AUROC ↓)', fontsize=9)
        else:
            ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=8)

    cax = fig.add_subplot(gs[1, K])
    cb = fig.colorbar(cbar_im, cax=cax)
    cb.set_label(f'Attention contrast ({pos_label} − {neg_label})', fontsize=9)
    cb.ax.tick_params(labelsize=7.5)

    fig.suptitle(ds_label, fontsize=12, fontweight='bold', y=0.96)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


# ====================================================================
# D. Cz-only morphology (with caching)
# ====================================================================

def _build_morphology_cache(ds_label, ds_dir, var_dir, det_idx, cache_path):
    """Compute per-subject TP/FN/TN/FP grand-mean waveforms at the
    detection channel from epoched_fif + predictions, save to npz.

    Uses the same event filtering as 06_gen_analysis.py / training:
    keep epochs whose event name matches pos_key or neg_key (or
    label_groups if defined), drop everything else."""
    import mne

    cfg = json.load(open(DATASETS_DIR / ds_dir / 'dataset_config.json'))
    pos_key = cfg['label_map']['pos_key']
    neg_key = cfg['label_map']['neg_key']
    label_groups = cfg.get('label_groups')

    rdir = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / 'erpxttn_auto'
    res = json.load(open(rdir / 'results.json'))
    subjects = [r['test_subject'] for r in res['folds']]
    proto_windows_per_fold = {}
    sfreq = None
    n_times = None
    ch_names_all = None

    epoch_root = DATASETS_DIR / ds_dir / 'epoched_fif' / 'tmin0ms_tmax800ms' / var_dir
    if not epoch_root.exists():
        raise FileNotFoundError(
            f'epoched_fif missing for {ds_label}: {epoch_root}\n'
            'Run 03_preprocess.py first.'
        )

    per_subj_tp, per_subj_fn, per_subj_tn, per_subj_fp = {}, {}, {}, {}
    counts = {}

    for subj in subjects:
        # Load predictions
        pred = np.load(rdir / f'predictions_{subj}.npz')
        probs = pred['probs']
        labels = pred['labels']

        # Load epochs (concatenate runs/sessions) with event filtering
        subj_dir = epoch_root / subj
        fifs = sorted(subj_dir.rglob('*-epo.fif'))
        if not fifs:
            print(f'    {subj}: no FIFs, skipping')
            continue

        all_X, all_y = [], []
        ch_names = None
        for f in fifs:
            ep = mne.read_epochs(str(f), preload=True, verbose=False)
            if ch_names is None:
                ch_names = ep.ch_names
            evid = ep.event_id
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
            X = ep.get_data()
            codes = ep.events[:, 2]
            mask = np.isin(codes, list(keep_ids))
            all_X.append(X[mask])
            y_local = np.array([1 if c in pos_ids else 0 for c in codes[mask]])
            all_y.append(y_local)

        if not all_X:
            print(f'    {subj}: no matching events, skipping')
            continue
        X = np.concatenate(all_X, axis=0)  # (N, C, T)
        y_loaded = np.concatenate(all_y, axis=0)
        X_uV = X * 1e6

        if X_uV.shape[0] != len(labels):
            print(f'    {subj}: epoch count mismatch ({X_uV.shape[0]} vs '
                  f'predictions {len(labels)}), skipping')
            continue
        if not np.array_equal(y_loaded, labels):
            print(f'    {subj}: label order mismatch, skipping')
            continue

        if n_times is None:
            n_times = X_uV.shape[2]
            ch_names_all = ch_names

        # Pull detection channel slice
        det_signal = X_uV[:, det_idx, :]  # (N, T)
        preds = (probs >= 0.5).astype(int)
        tp = (labels == 1) & (preds == 1)
        fn = (labels == 1) & (preds == 0)
        tn = (labels == 0) & (preds == 0)
        fp = (labels == 0) & (preds == 1)

        def _grandmean(mask):
            if mask.sum() == 0:
                return None, 0
            return det_signal[mask].mean(axis=0), int(mask.sum())

        tp_m, n_tp = _grandmean(tp)
        fn_m, n_fn = _grandmean(fn)
        tn_m, n_tn = _grandmean(tn)
        fp_m, n_fp = _grandmean(fp)

        per_subj_tp[subj] = tp_m
        per_subj_fn[subj] = fn_m
        per_subj_tn[subj] = tn_m
        per_subj_fp[subj] = fp_m
        counts[subj] = {'tp': n_tp, 'fn': n_fn, 'tn': n_tn, 'fp': n_fp}

        # Capture prototype windows + sfreq from any subject (consistent across folds)
        if (rdir / f'prototypes_{subj}.npz').exists():
            p = np.load(rdir / f'prototypes_{subj}.npz')
            proto_windows_per_fold[subj] = [tuple(w) for w in p['proto_windows_ms']]
            if sfreq is None:
                sfreq = float(p['sfreq'])

        print(f'    {subj}: TP={n_tp}, FN={n_fn}, TN={n_tn}, FP={n_fp}')

    if n_times is None:
        raise RuntimeError(f'No subjects processed successfully for {ds_label}')

    # Pack into arrays (subjects × T) — pad with NaN where missing
    def _pack(d_dict):
        out = []
        for s in subjects:
            v = d_dict.get(s)
            out.append(np.full(n_times, np.nan) if v is None else v)
        return np.array(out)
    ch_names = ch_names_all

    np.savez_compressed(
        cache_path,
        subjects=np.array(subjects),
        det_channel_name=ch_names[det_idx],
        ch_names=np.array(ch_names),
        det_idx=det_idx,
        sfreq=sfreq,
        tp=_pack(per_subj_tp),
        fn=_pack(per_subj_fn),
        tn=_pack(per_subj_tn),
        fp=_pack(per_subj_fp),
        counts=np.array([(s, counts[s]['tp'], counts[s]['fn'],
                          counts[s]['tn'], counts[s]['fp'])
                         for s in subjects], dtype=object),
        proto_windows=np.array([proto_windows_per_fold.get(s, [])
                                for s in subjects], dtype=object),
    )
    print(f'  cached morphology to {cache_path}')


def fig_morphology_cz(ds_label, ds_dir, var_dir, det_idx, out_path):
    cache_path = CACHE_DIR / f'{ds_dir}_3ch_Cz.npz'
    if not cache_path.exists():
        print(f'  building cache for {ds_label}...')
        _build_morphology_cache(ds_label, ds_dir, var_dir, det_idx, cache_path)

    d = np.load(cache_path, allow_pickle=True)
    sfreq = float(d['sfreq'])
    det_name = str(d['det_channel_name'])
    tp = d['tp']  # (n_subj, T)
    fn = d['fn']
    tn = d['tn']
    fp = d['fp']
    subjects = list(d['subjects'])

    T = tp.shape[1]
    time_ms = np.arange(T) / sfreq * 1000.0

    # Aggregate across subjects (mean ± SEM across subjects)
    def _agg(arr):
        valid = ~np.isnan(arr[:, 0])
        n = int(valid.sum())
        if n == 0:
            return None, None, 0
        m = np.nanmean(arr, axis=0)
        s = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(T)
        return m, s, n

    tp_m, tp_s, n_tp = _agg(tp)
    fn_m, fn_s, n_fn = _agg(fn)
    tn_m, tn_s, n_tn = _agg(tn)
    fp_m, fp_s, n_fp = _agg(fp)

    # Total TP/FN/TN/FP trial counts across subjects
    counts = d['counts']
    total = {'tp': 0, 'fn': 0, 'tn': 0, 'fp': 0}
    for row in counts:
        _, t_tp, t_fn, t_tn, t_fp = row
        total['tp'] += int(t_tp); total['fn'] += int(t_fn)
        total['tn'] += int(t_tn); total['fp'] += int(t_fp)

    # Pull the windows from the first subject's prototypes if available
    pwins = d['proto_windows']
    proto_windows = pwins[0] if len(pwins) > 0 else []

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)

    for ax_idx, (a_m, a_s, a_label, a_color, b_m, b_s, b_label, b_color, panel_name) in enumerate([
        (tp_m, tp_s, f'TP (hit, n={n_tp} subj)', '#197b30',
         fn_m, fn_s, f'FN (miss, n={n_fn} subj)', '#b22222', 'Error class'),
        (tn_m, tn_s, f'TN (correct reject, n={n_tn} subj)', '#1f4e79',
         fp_m, fp_s, f'FP (false alarm, n={n_fp} subj)', '#d97706', 'Correct class'),
    ]):
        ax = axes[ax_idx]
        # Prototype window shading
        for w in proto_windows:
            ax.axvspan(w[0], w[1], color='gray', alpha=0.10, zorder=1)
        if a_m is not None:
            ax.plot(time_ms, a_m, color=a_color, lw=1.8, label=a_label, zorder=3)
            ax.fill_between(time_ms, a_m - a_s, a_m + a_s,
                            color=a_color, alpha=0.20, lw=0, zorder=2)
        if b_m is not None:
            ax.plot(time_ms, b_m, color=b_color, lw=1.8, label=b_label, zorder=3)
            ax.fill_between(time_ms, b_m - b_s, b_m + b_s,
                            color=b_color, alpha=0.20, lw=0, zorder=2)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel(f'{panel_name}\namplitude (µV)', fontsize=9)
        ax.legend(loc='best', fontsize=7.5, frameon=False)
        ax.tick_params(axis='both', labelsize=8)

    axes[1].set_xlabel('Time (ms)', fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    fig.suptitle(
        f'{ds_label} — outcome-conditioned grand averages at {det_name} '
        f'(3-channel)\n'
        f'totals: TP={total["tp"]}, FN={total["fn"]}, '
        f'TN={total["tn"]}, FP={total["fp"]}',
        fontsize=10, y=0.98,
    )
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  wrote {out_path}')


# ====================================================================
# D2. All-datasets TP/FN/TN/FP morphology grid (supplementary)
# ====================================================================

def _ensure_morphology_cache(ds_label, ds_dir, var_dir, det_idx):
    """Build per-subject TP/FN/TN/FP cache for one dataset, lazily running
    preprocessing first if needed. After caching, deletes the epoched_fif
    tree to keep disk use bounded.

    Returns True if the cache exists at the end of the call.
    """
    cache_path = CACHE_DIR / f'{ds_dir}_3ch_Cz.npz'
    if cache_path.exists():
        return True

    raw_fif = DATASETS_DIR / ds_dir / 'raw_fif'
    if not raw_fif.exists() or not any(raw_fif.iterdir()):
        print(f'  [skip] {ds_label}: no raw_fif/ in datasets/{ds_dir}; '
              'provide source data and rerun.')
        return False

    epoch_root = (DATASETS_DIR / ds_dir / 'epoched_fif'
                  / 'tmin0ms_tmax800ms' / var_dir)
    cleanup = False
    if not epoch_root.exists():
        # Parse channel preset and resample rate from the variant dir name.
        # Format: 'noref_<channels>_[rs<num>_]<filter>_bp-<l>-<h>'
        toks = var_dir.split('_')
        if toks[0] != 'noref':
            print(f'  [skip] {ds_label}: unexpected variant prefix '
                  f'{var_dir!r}.')
            return False
        rest = toks[1:]
        resample = None
        rs_i = None
        for i, t in enumerate(rest):
            if t.startswith('rs') and t[2:].isdigit():
                resample = int(t[2:])
                rs_i = i
                break
        if rs_i is not None:
            ch_tokens = rest[:rs_i]
        else:
            ch_tokens = []
            for t in rest:
                if t in ('iir', 'fir'):
                    break
                ch_tokens.append(t)
        channels = '_'.join(ch_tokens)
        if not channels:
            print(f'  [skip] {ds_label}: could not parse channels from '
                  f'{var_dir!r}.')
            return False

        cmd = [
            'python', str(REPO / '03_preprocess.py'),
            '--dataset', f'datasets/{ds_dir}',
            '--channels', channels,
            '--reference', 'none',
        ]
        if resample is not None:
            cmd += ['--resample', str(resample)]
        print(f'  preprocessing {ds_label}: {" ".join(cmd)}')
        import subprocess
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            print(f'  [skip] {ds_label}: preprocess failed (rc={rc}).')
            return False
        cleanup = True

    try:
        _build_morphology_cache(ds_label, ds_dir, var_dir, det_idx, cache_path)
    except Exception as e:
        print(f'  [skip] {ds_label}: cache build failed: {e}')
        return False

    if cleanup:
        import shutil
        epoch_top = DATASETS_DIR / ds_dir / 'epoched_fif'
        if epoch_top.exists():
            print(f'  cleaning up {epoch_top}')
            shutil.rmtree(epoch_top)
    return cache_path.exists()


def fig_morphology_grid_supp(out_path):
    """3x3 grid (Table 2 order): TP/FN (top) and TN/FP (bottom) grand-mean
    waveforms at the detection channel for each dataset. No SEM ribbons.

    Datasets without an available cache render as a placeholder panel —
    the script can be re-run after the missing source data is provided.
    """
    color_tp, color_fn = '#197b30', '#b22222'
    color_tn, color_fp = '#1f4e79', '#d97706'

    n_rows, n_cols = 3, 3

    fig = plt.figure(figsize=(7.5, 9.5))
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.55, wspace=0.38,
                             left=0.07, right=0.99, top=0.93, bottom=0.04)

    # Local row order: ERN, HRI ErrP, BNCI ErrP, LRP, then DATASETS order —
    # matching fig_prototypes_all_datasets so the two supp figures align.
    row_order = ['ERN', 'HRI ErrP', 'BNCI ErrP', 'LRP',
                 'N170', 'P300', 'N2pc', 'MMN', 'N400']
    by_name = {row[0]: row for row in DATASETS}
    datasets_ordered = [by_name[n] for n in row_order]

    for idx, (ds_label, ds_dir, c3, _cf, _k3, _kf, det_idx) in enumerate(datasets_ordered):
        r, c = idx // n_cols, idx % n_cols
        ok = _ensure_morphology_cache(ds_label, ds_dir, c3, det_idx)

        inner = outer[r, c].subgridspec(2, 1, hspace=0.12)
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

        if not ok:
            for ax in (ax_top, ax_bot):
                ax.text(0.5, 0.5, '(data unavailable)',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=8.5, color='gray', style='italic')
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color('lightgray')
            ax_top.set_title(ds_label, fontsize=9.5, pad=2)
            continue

        cache_path = CACHE_DIR / f'{ds_dir}_3ch_Cz.npz'
        d = np.load(cache_path, allow_pickle=True)
        sfreq = float(d['sfreq'])
        det_name = str(d['det_channel_name'])
        T = d['tp'].shape[1]
        time_ms = np.arange(T) / sfreq * 1000.0

        def _gm(arr):
            valid = ~np.isnan(arr[:, 0])
            if valid.sum() == 0:
                return None
            return np.nanmean(arr, axis=0)

        tp_m = _gm(d['tp']); fn_m = _gm(d['fn'])
        tn_m = _gm(d['tn']); fp_m = _gm(d['fp'])

        ax_top.axhline(0, color='gray', lw=0.4, ls='--', zorder=1)
        if tp_m is not None:
            ax_top.plot(time_ms, tp_m, color=color_tp, lw=1.2, zorder=3)
        if fn_m is not None:
            ax_top.plot(time_ms, fn_m, color=color_fn, lw=1.2, zorder=3)

        ax_bot.axhline(0, color='gray', lw=0.4, ls='--', zorder=1)
        if tn_m is not None:
            ax_bot.plot(time_ms, tn_m, color=color_tn, lw=1.2, zorder=3)
        if fp_m is not None:
            ax_bot.plot(time_ms, fp_m, color=color_fp, lw=1.2, zorder=3)

        ax_top.set_title(f'{ds_label} — {det_name}', fontsize=9.5, pad=2)

        for ax in (ax_top, ax_bot):
            ax.tick_params(axis='both', labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        if r == n_rows - 1:
            ax_bot.set_xlabel('Time (ms)', fontsize=8)
        if c == 0:
            ax_top.set_ylabel('µV', fontsize=8)
            ax_bot.set_ylabel('µV', fontsize=8)

    # Shared legend at bottom
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=color_tp, lw=1.6, label='TP (hit)'),
        Line2D([0], [0], color=color_fn, lw=1.6, label='FN (miss)'),
        Line2D([0], [0], color=color_tn, lw=1.6, label='TN (correct reject)'),
        Line2D([0], [0], color=color_fp, lw=1.6, label='FP (false alarm)'),
    ]
    fig.legend(handles=handles, loc='upper center', frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.99), ncol=4,
               handletextpad=0.5, columnspacing=1.6)

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  wrote {out_path}')


# ====================================================================
# E. All-datasets prototype grid (detection channel only)
# ====================================================================

PROTO_COLOR_PALETTE = ['#e67e22', '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#1abc9c']


def _proto_names_for(ds_dir, K):
    """Pad/truncate dataset config's proto_names to length K."""
    cfg = json.load(open(DATASETS_DIR / ds_dir / 'dataset_config.json'))
    names = list(cfg.get('proto_names', []))
    while len(names) < K:
        names.append(f'Proto-{len(names) + 1}')
    return names[:K]


def _proto_fold_names(proto_raw, proto_windows, det_idx, sfreq, ds_dir=None):
    """Return prototype names. Prefer dataset_config.proto_names when its
    length matches the actual K; otherwise fall back to the auto-mode
    polarity-derived names (P1, P2, ... / N1, N2, ...)."""
    K = proto_raw.shape[0]
    if ds_dir is not None:
        cfg = json.load(open(DATASETS_DIR / ds_dir / 'dataset_config.json'))
        cfg_names = list(cfg.get('proto_names', []))
        if len(cfg_names) == K:
            return cfg_names
    pos_count = neg_count = 0
    names = []
    for k in range(K):
        s_ms, e_ms = proto_windows[k]
        s_samp = int(round(s_ms / 1000 * sfreq))
        e_samp = int(round(e_ms / 1000 * sfreq))
        seg = proto_raw[k, det_idx, s_samp:e_samp]
        peak_val = seg[int(np.argmax(np.abs(seg)))] if len(seg) > 0 else 0
        if peak_val >= 0:
            pos_count += 1
            names.append(f'P{pos_count}')
        else:
            neg_count += 1
            names.append(f'N{neg_count}')
    return names


def _load_subject_epochs_simple(cfg, var_dir, subject, ds_dir):
    """Load a subject's filtered epochs (ordered to match predictions)."""
    import mne
    pos_key = cfg['label_map']['pos_key']
    neg_key = cfg['label_map']['neg_key']
    label_groups = cfg.get('label_groups')
    epoch_root = DATASETS_DIR / ds_dir / 'epoched_fif' / 'tmin0ms_tmax800ms' / var_dir
    subj_dir = epoch_root / subject
    fifs = sorted(subj_dir.rglob('*-epo.fif'))
    all_X, all_y = [], []
    ch_names = None
    for f in fifs:
        ep = mne.read_epochs(str(f), preload=True, verbose=False)
        if ch_names is None:
            ch_names = ep.ch_names
        evid = ep.event_id
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
        X = ep.get_data()
        codes = ep.events[:, 2]
        mask = np.isin(codes, list(keep_ids))
        all_X.append(X[mask])
        all_y.append(np.array([1 if c in pos_ids else 0 for c in codes[mask]]))
    return np.concatenate(all_X, 0), np.concatenate(all_y, 0), ch_names


PROTO_COLORS_TPTN = ['#e67e22', '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#1abc9c']


def fig_tp_tn_two_subjects(ds_label, ds_dir, var_dir, det_idx,
                           subjects, out_path, conf='highconf'):
    """Combined TP/TN routing figure for two subjects with a shared
    prototype reference at the top. Mirrors the per-subject layout in
    05_gen_figures.py but stacks two subjects vertically.

    subjects: list of two subject IDs, e.g. ['sub-03','sub-10'].
    conf: 'highconf' (max-confidence) or 'median'.
    """
    import matplotlib.gridspec as gridspec
    cfg = json.load(open(DATASETS_DIR / ds_dir / 'dataset_config.json'))
    pos_label = cfg['label_map']['pos_key'].replace('_', ' ').title()
    neg_label = cfg['label_map']['neg_key'].replace('_', ' ').title()
    det_name = cfg.get('detection_channel', 'Cz')

    rdir = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / 'erpxttn_auto'

    # Load shared prototypes from the first subject (windows + waveforms
    # are per-fold, but the *reference* panel uses one subject's prototypes
    # — same convention as the per-subject figures).
    ref_subject = subjects[0]
    p_ref = np.load(rdir / f'prototypes_{ref_subject}.npz')
    proto_raw_ref = p_ref['proto_raw']
    proto_windows_ref = [tuple(w) for w in p_ref['proto_windows_ms']]
    sfreq = float(p_ref['sfreq'])
    K_ref = proto_raw_ref.shape[0]
    T = proto_raw_ref.shape[2]
    time_ms = np.arange(T) / sfreq * 1000
    fold_names_ref = _proto_fold_names(proto_raw_ref, proto_windows_ref,
                                       det_idx, sfreq, ds_dir=ds_dir)

    # Load per-subject data
    per_sub = {}
    for s in subjects:
        preds = np.load(rdir / f'predictions_{s}.npz')
        attn = np.load(rdir / f'attention_{s}.npz')['attention_weights']
        proto = np.load(rdir / f'prototypes_{s}.npz')
        proto_windows = [tuple(w) for w in proto['proto_windows_ms']]
        proto_raw = proto['proto_raw']
        X_raw, y_raw, ch_names = _load_subject_epochs_simple(
            cfg, var_dir, s, ds_dir)
        if not np.array_equal(y_raw, preds['labels']):
            print(f'  WARN: {s} label order mismatch; epoch labels may be misaligned')
        probs = preds['probs']
        labels = preds['labels']
        tp_mask = (labels == 1) & (probs >= 0.5)
        tn_mask = (labels == 0) & (probs < 0.5)
        tp_idx = np.where(tp_mask)[0]
        tn_idx = np.where(tn_mask)[0]
        if conf == 'highconf':
            tp_pick = tp_idx[np.argmax(probs[tp_idx])]
            tn_pick = tn_idx[np.argmin(probs[tn_idx])]
        else:
            tp_sorted = tp_idx[np.argsort(probs[tp_idx])]
            tn_sorted = tn_idx[np.argsort(probs[tn_idx])]
            tp_pick = tp_sorted[len(tp_sorted) // 2]
            tn_pick = tn_sorted[len(tn_sorted) // 2]
        per_sub[s] = {
            'attn': attn,  # (N_trials, H, N_patches, K)
            'proto_raw': proto_raw,
            'proto_windows': proto_windows,
            'X_raw': X_raw,
            'tp': tp_pick, 'tn': tn_pick,
            'p_tp': float(probs[tp_pick]),
            'p_tn': float(probs[tn_pick]),
            'auroc': float(preds['auroc']),
            'fold_names': _proto_fold_names(proto_raw, proto_windows,
                                            det_idx, sfreq, ds_dir=ds_dir),
        }

    n_subj = len(subjects)
    # Layout: 1 (proto ref) + n_subj × 2 (cz, attn) rows
    fig = plt.figure(figsize=(15, 5.5 + 5.5 * n_subj))
    height_ratios = [0.85] + [1.4, 1.05] * n_subj
    gs = gridspec.GridSpec(
        1 + 2 * n_subj, 2,
        height_ratios=height_ratios,
        hspace=0.65, wspace=0.18,
        figure=fig,
    )

    # ----- Row 0: Prototype reference (spans both columns) -----
    ax_proto = fig.add_subplot(gs[0, :])
    proto_cz = proto_raw_ref[:, det_idx, :]
    for k in range(K_ref):
        s_ms, e_ms = proto_windows_ref[k]
        ax_proto.axvspan(s_ms, e_ms, color=PROTO_COLORS_TPTN[k],
                         alpha=0.20, zorder=1)
    for k in range(K_ref):
        s_ms, e_ms = proto_windows_ref[k]
        s_samp = int(round(s_ms / 1000 * sfreq))
        e_samp = int(round(e_ms / 1000 * sfreq))
        ax_proto.plot(time_ms, proto_cz[k], color=PROTO_COLORS_TPTN[k],
                      lw=0.6, alpha=0.3, zorder=2)
        ax_proto.plot(time_ms[s_samp:e_samp], proto_cz[k, s_samp:e_samp],
                      color=PROTO_COLORS_TPTN[k], lw=2.5, zorder=3,
                      label=f'{fold_names_ref[k]} ({s_ms:.0f}–{e_ms:.0f} ms)')
    ax_proto.axhline(0, color='gray', lw=0.5, ls='--')
    ax_proto.set_xlim(0, time_ms[-1])
    ax_proto.set_ylabel(f'Prototype {det_name}\n(z-score)', fontsize=12)
    ax_proto.set_title(f'Diff-Wave Prototypes ({det_name} channel) — {ds_label}',
                       fontsize=14, fontweight='bold')
    ax_proto.legend(fontsize=11, loc='upper right', ncol=K_ref)
    ax_proto.tick_params(axis='both', labelsize=10)

    # ----- Subject blocks -----
    for si, s in enumerate(subjects):
        d = per_sub[s]
        K = d['proto_raw'].shape[0]
        N_patches = d['attn'].shape[2]
        patch_width = T // N_patches
        patch_centers_ms = (np.arange(N_patches) * patch_width
                            + patch_width / 2) / sfreq * 1000

        # Mean over heads → (N_trials, N_patches, K)
        attn_h = d['attn'].mean(axis=1)
        tp_attn = attn_h[d['tp']]
        tn_attn = attn_h[d['tn']]
        tp_cz = d['X_raw'][d['tp'], det_idx, :] * 1e6
        tn_cz = d['X_raw'][d['tn'], det_idx, :] * 1e6

        ymax_cz = max(np.abs(tp_cz).max(), np.abs(tn_cz).max()) * 1.15
        ymax_attn = max(tp_attn.max(), tn_attn.max()) * 1.1

        # Subject header text above the row of axes
        # Use a "phantom" axis for the title bar
        # Actually, place suptitle using fig.text at y based on row position
        # Compute approximate y of this subject's first row top
        # Easier: just use ax title prefixed with subject

        trials = [
            (tp_cz, tp_attn, d['p_tp'], f'{pos_label} trial (TP)', 0),
            (tn_cz, tn_attn, d['p_tn'], f'{neg_label} trial (TN)', 1),
        ]
        cz_row = 1 + 2 * si
        attn_row = 2 + 2 * si

        for cz, trial_attn, prob, title, col in trials:
            ax1 = fig.add_subplot(gs[cz_row, col])
            for p in range(N_patches):
                s_samp = p * patch_width
                e_samp = (p + 1) * patch_width
                s_t = time_ms[s_samp]
                e_t = time_ms[min(e_samp, T - 1)]
                for k in range(K):
                    w = trial_attn[p, k]
                    if w > 0.02:
                        ax1.axvspan(s_t, e_t, color=PROTO_COLORS_TPTN[k],
                                    alpha=float(w) * 0.35, zorder=1, lw=0)
            for p in range(N_patches):
                s_samp = p * patch_width
                e_samp = min((p + 1) * patch_width + 1, T)
                dominant_k = int(np.argmax(trial_attn[p, :]))
                ax1.plot(time_ms[s_samp:e_samp], cz[s_samp:e_samp],
                         color=PROTO_COLORS_TPTN[dominant_k], lw=2.2,
                         zorder=3, solid_capstyle='round')
            ax1.axhline(0, color='gray', lw=0.5, ls='--', zorder=2)
            ttl = f'{title}\np({pos_label.lower()}) = {prob:.3f}'
            if col == 0:
                ttl = f'{s}  (AUROC = {d["auroc"]:.3f})\n' + ttl
            ax1.set_title(ttl, fontsize=13, fontweight='bold')
            ax1.set_ylabel(f'{det_name} amplitude (µV)', fontsize=12)
            ax1.set_xlim(0, time_ms[-1])
            ax1.set_ylim(-ymax_cz, ymax_cz)
            ax1.tick_params(axis='both', labelsize=10)

            ax2 = fig.add_subplot(gs[attn_row, col])
            for k in range(K):
                ax2.plot(patch_centers_ms, trial_attn[:, k],
                         color=PROTO_COLORS_TPTN[k], lw=1.8,
                         label=d['fold_names'][k] if col == 0 else None,
                         zorder=3)
            ax2.set_xlabel('Time (ms)', fontsize=12)
            ax2.set_ylabel('Attention weight', fontsize=12)
            ax2.set_ylim(0, ymax_attn)
            ax2.set_xlim(0, time_ms[-1])
            ax2.tick_params(axis='both', labelsize=10)
            if col == 0:
                ax2.legend(fontsize=11, loc='upper right')

    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def fig_prototypes_single_dataset(ds_label, ds_dir, var_dir, det_idx,
                                  out_path):
    """Single-row prototype figure for one dataset at the detection channel.
    Mirrors fig_prototypes.png from 05_gen_figures.py but with C=1
    (detection channel only) and a horizontal layout.
    """
    rdir = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / 'erpxttn_auto'
    proto_files = sorted(rdir.glob('prototypes_*.npz'))

    all_protos = {}
    per_fold_windows = {}
    sfreq = None
    for f in proto_files:
        subj = f.stem.replace('prototypes_', '')
        d = np.load(f)
        all_protos[subj] = d['proto_raw'][:, det_idx, :]
        per_fold_windows[subj] = [tuple(w) for w in d['proto_windows_ms']]
        if sfreq is None:
            sfreq = float(d['sfreq'])
    subjects = sorted(all_protos.keys(), key=lambda s: int(s.split('-')[1]))
    fold_K = {s: all_protos[s].shape[0] for s in subjects}
    K = max(fold_K.values())
    windows = []
    for k in range(K):
        wks = [per_fold_windows[s][k] for s in subjects if fold_K[s] > k]
        windows.append(tuple(np.round(np.mean(wks, axis=0), 1)))
    T = next(iter(all_protos.values())).shape[1]
    time_ms = np.arange(T) / sfreq * 1000
    proto_names = _proto_names_for(ds_dir, K)
    n_subj = len(subjects)

    trace_alpha = 0.65
    span_alpha = 0.04

    fig, axes = plt.subplots(1, K, figsize=(4.0 * K, 3.2),
                             sharex=True, squeeze=False)
    axes = axes[0]
    for k in range(K):
        ax = axes[k]
        color = PROTO_COLOR_PALETTE[k % len(PROTO_COLOR_PALETTE)]
        s_ms, e_ms = windows[k]
        k_subjects = [s for s in subjects if fold_K[s] > k]
        traces = np.array([all_protos[s][k] for s in k_subjects])

        for ti, t in enumerate(traces):
            ax.plot(time_ms, t, color=color, alpha=trace_alpha, lw=1.2,
                    zorder=2)
            fw = per_fold_windows[k_subjects[ti]]
            ax.axvspan(fw[k][0], fw[k][1], color=color, alpha=span_alpha,
                       zorder=1)

        mean = traces.mean(0)
        std = traces.std(0)
        ax.fill_between(time_ms, mean - std, mean + std,
                        color=color, alpha=0.15, zorder=3)
        ax.plot(time_ms, mean, color='black', lw=3.2, alpha=0.35, zorder=4)
        ax.plot(time_ms, mean, color=color, lw=2.2, zorder=5)
        ax.axvspan(s_ms, e_ms, color='gray', alpha=0.15, zorder=1)
        ax.axhline(0, color='gray', lw=0.6, ls='--', zorder=2)

        n_k = len(k_subjects)
        count_note = f' [n={n_k}]' if n_k < n_subj else ''
        ax.set_ylabel(
            f'{proto_names[k]} ({s_ms:.0f}–{e_ms:.0f} ms){count_note}\n(z-score)',
            fontsize=10,
        )
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def fig_prototypes_all_datasets():
    """One-figure grid of detection-channel prototypes for all 9 datasets,
    3-channel runs. Mirrors the per-fold/mean rendering of
    fig_prototypes.png from 05_gen_figures.py, with one row per dataset
    and one column per prototype slot (max K across datasets).

    Row ordering deviates from DATASETS / Table 2: LRP is moved to follow
    the ErrP block (HRI, BNCI), grouping the response-locked datasets
    visually rather than splitting them with the stimulus-locked rows.
    """

    # Local row order: ERN, HRI ErrP, BNCI ErrP, LRP, then DATASETS order.
    row_order = ['ERN', 'HRI ErrP', 'BNCI ErrP', 'LRP',
                 'N170', 'P300', 'N2pc', 'MMN', 'N400']
    by_name = {row[0]: row for row in DATASETS}
    datasets_ordered = [by_name[n] for n in row_order]

    # Load all data first to determine grid size
    per_ds = []
    K_max = 0
    for name, ds_dir, c3, cf, k3, kf, det_idx in datasets_ordered:
        rdir = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / c3 / 'erpxttn_auto'
        proto_files = sorted(rdir.glob('prototypes_*.npz'))
        if not proto_files:
            continue
        all_protos = {}        # subj -> (K_subj, T) at det channel
        per_fold_windows = {}  # subj -> [(s,e), ...]
        sfreq = None
        for f in proto_files:
            subj = f.stem.replace('prototypes_', '')
            d = np.load(f)
            proto = d['proto_raw']  # (K_subj, C, T)
            all_protos[subj] = proto[:, det_idx, :]
            per_fold_windows[subj] = [tuple(w) for w in d['proto_windows_ms']]
            if sfreq is None:
                sfreq = float(d['sfreq'])
        subjects = sorted(all_protos.keys(), key=lambda s: int(s.split('-')[1]))
        fold_K = {s: all_protos[s].shape[0] for s in subjects}
        K = max(fold_K.values())
        K_max = max(K_max, K)
        # Mean window per slot
        windows = []
        for k in range(K):
            wks = [per_fold_windows[s][k] for s in subjects if fold_K[s] > k]
            windows.append(tuple(np.round(np.mean(wks, axis=0), 1)))
        T = next(iter(all_protos.values())).shape[1]
        time_ms = np.arange(T) / sfreq * 1000
        per_ds.append({
            'name': name, 'subjects': subjects, 'protos': all_protos,
            'per_fold_windows': per_fold_windows, 'fold_K': fold_K,
            'K': K, 'windows': windows, 'time_ms': time_ms,
            'proto_names': _proto_names_for(ds_dir, K),
            'n_subj': len(subjects),
        })

    n_rows = len(per_ds)
    n_cols = K_max  # 4

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 1.55 * n_rows),
        sharex=True, squeeze=False,
    )

    for ri, ds in enumerate(per_ds):
        n_subj = ds['n_subj']
        K = ds['K']
        # Match alpha scaling from 05_gen_figures.py
        trace_alpha = max(0.08, min(0.25, 3.0 / n_subj))
        span_alpha = max(0.005, min(0.04, 0.5 / n_subj))

        for ci in range(n_cols):
            ax = axes[ri, ci]
            if ci >= K:
                ax.set_visible(False)
                continue
            color = PROTO_COLOR_PALETTE[ci % len(PROTO_COLOR_PALETTE)]
            s_ms, e_ms = ds['windows'][ci]
            k_subjects = [s for s in ds['subjects'] if ds['fold_K'][s] > ci]
            traces = np.array([ds['protos'][s][ci] for s in k_subjects])

            # Per-fold thin traces + per-fold window spans
            for ti, t in enumerate(traces):
                ax.plot(ds['time_ms'], t, color=color, alpha=trace_alpha,
                        lw=1.0, zorder=2)
                fw = ds['per_fold_windows'][k_subjects[ti]]
                ax.axvspan(fw[ci][0], fw[ci][1], color=color, alpha=span_alpha,
                           zorder=1)

            mean = traces.mean(0)
            std = traces.std(0)
            ax.fill_between(ds['time_ms'], mean - std, mean + std,
                            color=color, alpha=0.15, zorder=3)
            ax.plot(ds['time_ms'], mean, color='black', lw=2.6, alpha=0.35,
                    zorder=4)
            ax.plot(ds['time_ms'], mean, color=color, lw=1.8, zorder=5)
            # Mean window shading (gray, matches per-dataset figure)
            ax.axvspan(s_ms, e_ms, color='gray', alpha=0.15, zorder=1)
            ax.axhline(0, color='gray', lw=0.5, ls='--', zorder=2)

            n_k = len(k_subjects)
            count_note = f' [n={n_k}]' if n_k < n_subj else ''
            ax.set_ylabel(
                f'{ds["proto_names"][ci]} ({s_ms:.0f}–{e_ms:.0f} ms){count_note}\n(z-score)',
                fontsize=7.5,
            )
            ax.tick_params(axis='both', labelsize=7)
            if ri == n_rows - 1:
                ax.set_xlabel('Time (ms)', fontsize=8.5)

        # Dataset row label on the left, outside the first panel
        axes[ri, 0].annotate(
            ds['name'], xy=(-0.42, 0.5), xycoords='axes fraction',
            ha='center', va='center', fontsize=10, fontweight='bold',
            rotation=90,
        )

    fig.suptitle(
        'Difference-wave prototypes (LOSO folds) across datasets — '
        '3-channel, detection channel only',
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0.04, 0, 1, 0.98))
    out = OUT_DIR / 'fig_prototypes_all_datasets.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


# ====================================================================
# Main
# ====================================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    print('Tax-driver scatter...')
    fig_tax_drivers()

    print('TP↔FP / TP↔TN bar chart...')
    fig_tpfp_tptn()

    print('Routing combined figures (3-channel only)...')
    for name, ds_dir, c3, cf, k3, kf, det_idx in DATASETS:
        slug = ds_dir.replace('-', '_')
        out = OUT_DIR / f'fig_routing_{slug}_3ch.png'
        try:
            fig_routing_combined(name, ds_dir, c3, out)
        except Exception as e:
            import traceback
            print(f'  FAILED {name}: {e}')
            traceback.print_exc()

    print('HRI Cz-only morphology...')
    hri = next(r for r in DATASETS if r[1] == 'hri_errp_cursor')
    fig_morphology_cz(hri[0], hri[1], hri[2], hri[6],
                      OUT_DIR / 'fig_morphology_hri_errp_3ch_Cz.png')

    print('BNCI Cz-only morphology (supp replication)...')
    bnci = next(r for r in DATASETS if r[1] == 'bnci_errp_013-2015')
    fig_morphology_cz(bnci[0], bnci[1], bnci[2], bnci[6],
                      OUT_DIR / 'fig_morphology_bnci_errp_3ch_Cz.png')

    print('All-datasets prototype grid...')
    fig_prototypes_all_datasets()

    print('All-datasets morphology grid (supp)...')
    fig_morphology_grid_supp(OUT_DIR / 'fig_morphology_all_datasets_3ch.png')

    print('Done.')


if __name__ == '__main__':
    main()
