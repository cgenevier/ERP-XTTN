#!/usr/bin/env python3
"""07_gen_paper_figures.py — publication-grade figures for the extension.

Reads from analysis_summary.json + per-subject results files. Produces:

  paper_figures/fig_tax_drivers.png
  paper_figures/fig_tpfp_tptn.png
  paper_figures/fig_paired_heatmap.png   (Table 2 companion; HL Δ + BH-signed-rank)
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
from scipy.stats import spearmanr, wilcoxon

REPO = Path(__file__).resolve().parent
DATASETS_DIR = REPO / 'datasets'
OUT_DIR = REPO / 'paper_figures'
CACHE_DIR = OUT_DIR / 'morphology_cache'
ANALYSIS = REPO / 'analysis_summary.json'

# Results are stored per seed under <model>/seed-<N>/. Interpretability figures
# (routing, prototypes, morphology) use a single reference seed; the per-subject
# AUROC figure averages each subject over all seeds.
REF_SEED = 1


def _seed_dir(ds_dir, var_dir, model='erpxttn_peak'):
    """Reference-seed results dir (seed-REF_SEED, else lowest seed present)."""
    base = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / model
    ref = base / f'seed-{REF_SEED}'
    if ref.exists():
        return ref
    seeds = sorted(base.glob('seed-*'), key=lambda d: int(d.name.split('-')[1]))
    return seeds[0] if seeds else ref


def _persubj_auroc(ds_dir, var_dir, model):
    """{subject: seed-averaged LOSO AUROC} across all seed dirs for a model.

    For the two-factor peak model this is the TWO-FACTOR (routing+amplitude)
    per-subject AUROC — the headline — when recorded; baselines use test_auroc.
    """
    from collections import defaultdict
    base = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / model
    acc = defaultdict(list)
    for sd in sorted(base.glob('seed-*')):
        rj = sd / 'results.json'
        if not rj.exists():
            continue
        r = json.load(open(rj))
        tf = r.get('two_factor_auroc_per_subject')
        if tf:
            for s, a in tf.items():
                acc[s].append(float(a))
        else:
            for f in r.get('folds', []):
                acc[f['test_subject']].append(float(f['test_auroc']))
    return {s: float(np.mean(v)) for s, v in acc.items()}

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

    def _taxd(src, bl):
        # Tax mean_delta for a baseline; np.nan if that baseline isn't in yet.
        t = src.get('tax', {}).get(bl)
        return t['mean_delta'] if t and t.get('mean_delta') is not None else np.nan

    rows_full, rows_3ch = [], []
    for name, ds_dir, c3, cf, k3, kf, _ in DATASETS:
        cf_obj = combos[kf]
        c3_obj = combos[k3]
        for src, rows in [(cf_obj, rows_full), (c3_obj, rows_3ch)]:
            rows.append({
                'name': name,
                'tax_eeg':  _taxd(src, 'eegnet'),
                'tax_xdr':  _taxd(src, 'xdawn_rg'),
                'tax_def':  _taxd(src, 'eeg_deformer'),
                'tax_epmn': _taxd(src, 'epmn'),
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
        ('tax_eeg',  r'$\Delta$ vs EEGNet'),
        ('tax_def',  r'$\Delta$ vs EEG-Deformer'),
        ('tax_epmn', r'$\Delta$ vs EPMN'),
        ('tax_xdr',  r'$\Delta$ vs xDAWN+RG'),
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

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.5), sharey=True)

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

    # 'n = 9 per subset' — bottom-right corner of last panel, unobtrusive
    axes[-1].text(0.98, 0.02, 'n = 9 per subset',
                  transform=axes[-1].transAxes,
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
# A2. Interpretability gap vs SNR proxy (main-text "Option B")
# ====================================================================

# Baselines shown against the SNR proxy (colors match MODELS_FOR_AUROC).
_SNR_BASELINES = [
    ('EEGNet',       'eegnet',        '#2980b9', 'o'),
    ('xDAWN+RG',     'xdawn_rg',      '#7f8c8d', 's'),
    ('EEG-Deformer', 'eeg_deformer',  '#27ae60', '^'),
    ('EPMN',         'epmn',          '#8e44ad', 'D'),
]


def fig_snr_gap():
    """Per-dataset interpretability gap (Delta = baseline - ERP-XTTN) against the
    SNR proxy, one panel per montage. Shows the single robust association from
    the fuller predictor analysis (fig_tax_drivers / Table SXX): at full montage
    the gap to every baseline widens with SNR, whereas at 3 channels it is small
    and unrelated to SNR. Trend lines and Spearman rho are per baseline (n=9
    datasets; descriptive, not a significance test)."""
    from scipy.stats import spearmanr
    from matplotlib.lines import Line2D
    summary = load_analysis()
    combos = summary['combos']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    # combo-key indices in the DATASETS tuple: 4 = 3-channel, 5 = full montage.
    montages = [('3ch', '3-channel', 4), ('full', 'Full montage', 5)]

    for ax, (mkey, mlabel, var_idx) in zip(axes, montages):
        legend_rhos = []
        for blabel, model, color, marker in _SNR_BASELINES:
            xs, ys = [], []
            for row in DATASETS:
                key = row[var_idx]                # 3ch_combo_key / full_combo_key
                c = combos.get(key)
                if not c:
                    continue
                t = c.get('tax', {}).get(model)
                snr = c.get('snr_proxy', {}).get('mean_snr')
                if not t or t.get('mean_delta') is None or snr is None:
                    continue
                xs.append(snr)
                ys.append(t['mean_delta'])
            if len(xs) < 3:
                continue
            xs, ys = np.array(xs), np.array(ys)
            ax.scatter(xs, ys, s=42, color=color, marker=marker, alpha=0.85,
                       edgecolor='white', linewidths=0.6, zorder=3, label=blabel)
            # Trend line (least-squares) for a visual guide.
            b1, b0 = np.polyfit(xs, ys, 1)
            xr = np.array([xs.min(), xs.max()])
            ax.plot(xr, b0 + b1 * xr, color=color, lw=1.4, alpha=0.7, zorder=2)
            rho, _ = spearmanr(xs, ys)
            legend_rhos.append((blabel, color, rho))

        ax.axhline(0.0, color='0.3', lw=1.0, ls='--', zorder=1)
        ax.set_title(mlabel, fontsize=11)
        ax.set_xlabel('SNR proxy  (|diff-wave| / trial SD)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Per-baseline Spearman rho as a text legend.
        txt = '\n'.join([r'$\rho$=' + f'{r:+.2f}  {l}' for l, _, r in legend_rhos])
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=7.5, linespacing=1.4,
                bbox=dict(boxstyle='round', fc='white', ec='0.8', alpha=0.85))

    axes[0].set_ylabel(r'$\Delta$ AUROC  (baseline $-$ ERP-XTTN)', fontsize=9)
    handles = [Line2D([0], [0], marker=mk, lw=0, color=c, markersize=8,
                      markeredgecolor='white', label=l)
               for l, _, c, mk in _SNR_BASELINES]
    fig.legend(handles=handles, ncol=4, loc='lower center', frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Interpretability gap vs signal strength', fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = OUT_DIR / 'fig_snr_gap.png'
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
    rdir = _seed_dir(ds_dir, var_dir)
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


# Peak-unit routing palette (matches 05_gen_figures / the reference figure).
PEAK_PROTO_COLORS = ['#d1710a', '#2e7d32', '#1565c0', '#8e24aa', '#c62828', '#00838f']


def _peak_comp_names(proto_det, windows, sfreq):
    names, npos, nneg = [], 0, 0
    for k, (s_ms, e_ms) in enumerate(windows):
        s, e = int(s_ms / 1000 * sfreq), int(e_ms / 1000 * sfreq)
        seg = proto_det[k, s:e]
        v = seg[int(np.argmax(np.abs(seg)))] if len(seg) else 0.0
        if v >= 0:
            npos += 1; names.append(f'P{npos}')
        else:
            nneg += 1; names.append(f'N{nneg}')
    return names


def _class_labels(ds_dir):
    cfg = json.load(open(DATASETS_DIR / ds_dir / 'dataset_config.json'))
    lm = cfg.get('label_map', {})
    return (_format_class_label(lm.get('pos_key', 'error')),
            _format_class_label(lm.get('neg_key', 'correct')))


def fig_routing_combined(ds_label, ds_dir, var_dir, out_path, n_bins=40):
    """Peak-unit aggregate routing figure (peak-unit redesign; reads the
    self-contained routing_<subj>.npz dumps).

    Panel A: per-prototype routed-attention timecourse — mean attention a[.,k]
    over detected peaks binned by peak-centre latency, error vs correct; each
    prototype's curve should rise inside its shaded component window.
    Panel B: per-subject heatmap of the class-difference grounded contribution
    (sum_k a.m) by latency.
    """
    import matplotlib.gridspec as gridspec
    rdir = _seed_dir(ds_dir, var_dir)
    paths = sorted(rdir.glob('routing_*.npz'))
    if not paths:
        print(f'  [skip] fig_routing_combined ({ds_label}): no routing dumps')
        return
    pos_label, neg_label = _class_labels(ds_dir)
    subj_ids = [p.stem.replace('routing_', '') for p in paths]
    z0 = np.load(paths[0], allow_pickle=True)
    windows = z0['proto_windows_ms']; sfreq = float(z0['sfreq'])
    chans = [str(c) for c in z0['channel_names']]
    det_name = str(z0['detection_channel']) if 'detection_channel' in z0.files else 'Cz'
    proto_raw = z0['proto_raw']; det = chans.index(det_name) if det_name in chans else 1
    K = z0['a'].shape[2]; T = proto_raw.shape[2]
    names = _peak_comp_names(proto_raw[:, det, :], windows, sfreq)
    edges = np.linspace(0, T, n_bins + 1)
    centers_ms = (0.5 * (edges[:-1] + edges[1:])) / sfreq * 1000

    a_sum = np.zeros((2, K, n_bins)); a_cnt = np.zeros((2, K, n_bins))
    heat = np.full((len(paths), n_bins), np.nan)
    for si, p in enumerate(paths):
        z = np.load(p, allow_pickle=True)
        a, m, mask, center, y = z['a'], z['m'], z['mask'], z['center'], z['labels']
        contrib = a * m
        bidx = np.clip(np.digitize(center, edges) - 1, 0, n_bins - 1)
        dc = np.zeros((2, n_bins)); dcnt = np.zeros((2, n_bins))
        for n in range(a.shape[0]):
            cls = int(y[n])
            for j in np.where(mask[n])[0]:
                b = bidx[n, j]
                a_sum[cls, :, b] += a[n, j]; a_cnt[cls, :, b] += 1
                dc[cls, b] += contrib[n, j].sum(); dcnt[cls, b] += 1
        with np.errstate(invalid='ignore'):
            heat[si] = (dc[1] / np.where(dcnt[1] > 0, dcnt[1], np.nan)
                        - dc[0] / np.where(dcnt[0] > 0, dcnt[0], np.nan))
    with np.errstate(invalid='ignore'):
        a_mean = a_sum / np.where(a_cnt > 0, a_cnt, np.nan)

    fig = plt.figure(figsize=(12, 8.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.9], hspace=0.32)
    axA = fig.add_subplot(gs[0])
    for k in range(K):
        c = PEAK_PROTO_COLORS[k % len(PEAK_PROTO_COLORS)]
        axA.axvspan(windows[k][0], windows[k][1], color=c, alpha=0.12)
        axA.plot(centers_ms, a_mean[1, k], color=c, lw=2.3, label=f'{names[k]} ({pos_label})')
        axA.plot(centers_ms, a_mean[0, k], color=c, lw=1.6, ls='--', alpha=0.8)
    axA.set_xlim(0, T / sfreq * 1000); axA.set_ylabel('Mean routed attention  a', fontsize=12)
    axA.set_title(f'Routing by peak latency - {ds_label}  (solid={pos_label}, dashed={neg_label})',
                  fontsize=13, fontweight='bold')
    axA.legend(fontsize=9, ncol=K, loc='upper right')
    axB = fig.add_subplot(gs[1])
    vmax = np.nanmax(np.abs(heat)) or 1.0
    im = axB.imshow(heat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    extent=[0, T / sfreq * 1000, len(subj_ids) - 0.5, -0.5], interpolation='nearest')
    axB.set_yticks(range(len(subj_ids))); axB.set_yticklabels(subj_ids, fontsize=6)
    axB.set_xlabel('Peak-centre latency (ms)', fontsize=12); axB.set_ylabel('Subject', fontsize=12)
    axB.set_title(f'Delta grounded contribution ({pos_label}-{neg_label}) by latency', fontsize=12, fontweight='bold')
    for k in range(K):
        axB.axvline(windows[k][0], color=PEAK_PROTO_COLORS[k % len(PEAK_PROTO_COLORS)], lw=0.8, alpha=0.5)
    fig.colorbar(im, ax=axB, fraction=0.03, pad=0.01, label='Delta a.m')
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  saved {Path(out_path).name}')


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

    rdir = _seed_dir(ds_dir, var_dir)
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


def _sel_conf(labels, probs, mode):
    err, cor = np.where(labels == 1)[0], np.where(labels == 0)[0]
    def cap(i):
        truth = 'Error' if labels[i] == 1 else 'Correct'
        ok = (labels[i] == 1) == (probs[i] >= 0.5)
        return f"{truth} p(err)={probs[i]:.2f} {'OK' if ok else 'MISS'}"
    if not len(err) or not len(cor):
        return None
    if mode == 'median':
        tp = err[np.argsort(probs[err])]; tn = cor[np.argsort(probs[cor])]
        L, R = tp[len(tp) // 2], tn[len(tn) // 2]
    else:
        L, R = err[np.argmax(probs[err])], cor[np.argmin(probs[cor])]
    return (int(L), cap(L)), (int(R), cap(R))


def _sig_panel(ax, z, det, tr, lab, T, sfreq, ymax):
    a, m, mask, center, bounds, X = z['a'], z['m'], z['mask'], z['center'], z['bounds'], z['X']
    time_ms = np.arange(T) / sfreq * 1000
    sig = X[tr, det]; cc = a[tr] * m[tr]; mm = m[tr]
    for j in np.where(mask[tr])[0]:
        lo, hi = int(bounds[tr, j, 0]), int(bounds[tr, j, 1])
        kbest = int(np.argmax(np.abs(cc[j]))); strong = np.abs(cc[j, kbest]) > 1e-3
        col = PEAK_PROTO_COLORS[kbest % len(PEAK_PROTO_COLORS)] if strong else '#9e9e9e'
        ax.axvspan(time_ms[max(0, lo)], time_ms[min(hi, T - 1)], color=col,
                   alpha=0.20 if strong else 0.06, lw=0, zorder=1)
        pc = int(min(center[tr, j], T - 1))
        ax.plot(time_ms[pc], sig[pc], 'v', ms=9, zorder=6,
                markerfacecolor=(col if mm[j, kbest] >= 0 else 'white'),
                markeredgecolor=col, markeredgewidth=1.4)
    ax.plot(time_ms, sig, color='#333', lw=1.8, zorder=3); ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_title(lab, fontsize=10, fontweight='bold'); ax.set_xlim(0, time_ms[-1]); ax.set_ylim(-ymax, ymax * 1.2)


def fig_tp_tn_two_subjects(ds_label, ds_dir, var_dir, det_idx, subjects, out_path, conf='highconf'):
    """Shared prototype strip + two subjects x (TP, TN) peak-routing signal rows
    (peak-unit redesign; reads routing_<subj>.npz)."""
    import matplotlib.gridspec as gridspec
    rdir = _seed_dir(ds_dir, var_dir)
    paths = [rdir / f'routing_{s}.npz' for s in subjects[:2]]
    if not all(p.exists() for p in paths):
        print(f'  [skip] fig_tp_tn_two_subjects ({ds_label}): missing routing dumps')
        return
    zs = [np.load(p, allow_pickle=True) for p in paths]
    z0 = zs[0]; windows = z0['proto_windows_ms']; sfreq = float(z0['sfreq'])
    chans = [str(c) for c in z0['channel_names']]
    det_name = str(z0['detection_channel']) if 'detection_channel' in z0.files else 'Cz'
    det = chans.index(det_name) if det_name in chans else 1
    proto_det = z0['proto_raw'][:, det, :]; T = z0['proto_raw'].shape[2]
    names = _peak_comp_names(proto_det, windows, sfreq); time_ms = np.arange(T) / sfreq * 1000
    mode = 'median' if conf == 'median' else 'highconf'
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.7, 1.0, 1.0], hspace=0.4, wspace=0.18)
    axp = fig.add_subplot(gs[0, :])
    for k in range(len(names)):
        c = PEAK_PROTO_COLORS[k % len(PEAK_PROTO_COLORS)]
        axp.axvspan(windows[k][0], windows[k][1], color=c, alpha=0.16)
        s, e = int(windows[k][0] / 1000 * sfreq), int(windows[k][1] / 1000 * sfreq)
        axp.plot(time_ms[s:e], proto_det[k, s:e], color=c, lw=2.5,
                 label=f'{names[k]} ({windows[k][0]:.0f}-{windows[k][1]:.0f} ms)')
    axp.axhline(0, color='gray', lw=0.5, ls='--'); axp.set_xlim(0, time_ms[-1])
    axp.set_title('Difference-Wave Prototypes', fontsize=13, fontweight='bold')
    axp.legend(fontsize=9, ncol=len(names), loc='upper right'); axp.set_ylabel(f'{det_name} (z)', fontsize=11)
    for row, (z, sid) in enumerate(zip(zs, subjects[:2])):
        sel = _sel_conf(z['labels'], z['probs'], mode)
        if sel is None:
            continue
        ymax = np.abs(z['X'][:, det]).max() * 1.1
        for col, (tr, lab) in enumerate(sel):
            ax = fig.add_subplot(gs[row + 1, col])
            _sig_panel(ax, z, det, tr, f'{sid} - {lab}', T, sfreq, ymax)
            if row == 1:
                ax.set_xlabel('Time (ms)', fontsize=11)
    fig.suptitle(f'Peak-Unit Routing - {ds_label} ({subjects[0]} vs {subjects[1]})',
                 fontsize=15, fontweight='bold', y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  saved {Path(out_path).name}')


def fig_prototypes_single_dataset(ds_label, ds_dir, var_dir, det_idx,
                                  out_path):
    """Single-row prototype figure for one dataset at the detection channel.
    Mirrors fig_prototypes.png from 05_gen_figures.py but with C=1
    (detection channel only) and a horizontal layout.
    """
    rdir = _seed_dir(ds_dir, var_dir)
    proto_files = sorted(rdir.glob('prototypes_*.npz'))

    all_protos = {}
    per_fold_windows = {}
    sfreq = None
    for f in proto_files:
        subj = f.stem.replace('prototypes_', '')
        d = np.load(f)
        all_protos[subj] = d['mf_template'][:, det_idx, :]
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
        rdir = _seed_dir(ds_dir, c3)
        proto_files = sorted(rdir.glob('prototypes_*.npz'))
        if not proto_files:
            continue
        all_protos = {}        # subj -> (K_subj, T) at det channel
        per_fold_windows = {}  # subj -> [(s,e), ...]
        sfreq = None
        for f in proto_files:
            subj = f.stem.replace('prototypes_', '')
            d = np.load(f)
            proto = d['mf_template']  # (K_subj, C, T)
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
# Per-subject AUROC (individual points, all models) — Editor request
# ====================================================================

MODELS_FOR_AUROC = [
    ('ERP-XTTN',     'erpxttn_peak',  '#c0392b'),
    ('EEGNet',       'eegnet',        '#2980b9'),
    ('xDAWN+RG',     'xdawn_rg',      '#7f8c8d'),
    ('EEG-Deformer', 'eeg_deformer',  '#27ae60'),
    ('EPMN',         'epmn',          '#8e44ad'),
]

# Axis labels for the faceted small-multiples (full method names).
_MODEL_SHORT = {
    'ERP-XTTN': 'ERP-XTTN', 'EEGNet': 'EEGNet', 'xDAWN+RG': 'xDAWN+RG',
    'EEG-Deformer': 'EEG-Deformer', 'EPMN': 'EPMN',
}


def fig_persubject_auroc(montage, out_name):
    """Per-subject LOSO AUROC (seed-averaged) for every model and dataset.

    One panel per dataset (3x3 small-multiples). Within each panel every model
    is shown as a box (median + IQR + whiskers) with the individual subject
    AUROCs jittered over it and the group mean marked by a diamond, so the
    reader sees the full cross-subject distribution rather than only the group
    means reported in the tables (Editor request). ``montage`` is '3ch' or
    'full'; a shared y-axis makes the across-component difficulty ordering
    directly comparable.
    """
    from matplotlib.lines import Line2D
    var_pick = 2 if montage == '3ch' else 3  # index into DATASETS tuple
    n_models = len(MODELS_FOR_AUROC)

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), sharey=True)
    axes = axes.ravel()
    any_data = False

    for di, row in enumerate(DATASETS):
        ax = axes[di]
        var_dir = row[var_pick]
        n_subj = 0
        for mi, (mlabel, model, color) in enumerate(MODELS_FOR_AUROC):
            pa = _persubj_auroc(row[1], var_dir, model)
            if not pa:
                continue
            any_data = True
            vals = np.array(list(pa.values()))
            n_subj = max(n_subj, len(vals))
            emph = (model == 'erpxttn_peak')

            # Box: median + IQR + whiskers (robust for n as small as 6).
            bp = ax.boxplot(
                vals, positions=[mi], widths=0.62, showfliers=False,
                patch_artist=True, zorder=2,
                medianprops=dict(color=color, lw=1.6),
                whiskerprops=dict(color=color, lw=0.9),
                capprops=dict(color=color, lw=0.9),
                boxprops=dict(edgecolor=color, lw=1.9 if emph else 1.0))
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.14)

            # Individual subjects, jittered.
            rng = np.random.RandomState(di * 100 + mi)
            jit = (rng.rand(len(vals)) - 0.5) * 0.42
            ax.scatter(np.full(len(vals), mi) + jit, vals, s=9, color=color,
                       alpha=0.55, edgecolor='none', zorder=3)

            # Group mean (matches the value tabulated in Table 2).
            ax.scatter([mi], [vals.mean()], marker='D', s=30, color='white',
                       edgecolor=color, linewidths=1.5, zorder=5)

        ax.axhline(0.5, ls='--', color='gray', lw=0.7, zorder=1)
        ax.set_title(f'{row[0]}' + (f'  (n={n_subj})' if n_subj else ''),
                     fontsize=10)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([_MODEL_SHORT[l] for l, _, _ in MODELS_FOR_AUROC],
                           rotation=30, ha='right', fontsize=7)
        ax.set_xlim(-0.6, n_models - 0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if di % 3 == 0:
            ax.set_ylabel('LOSO AUROC (per subject)', fontsize=9)

    if not any_data:
        plt.close(fig)
        print(f'  (no results yet for {montage}; skipping {out_name})')
        return

    axes[0].set_ylim(0.35, 1.02)
    for ax in axes[len(DATASETS):]:
        ax.axis('off')

    handles = [
        Line2D([0], [0], marker='D', lw=0, color='white', markeredgecolor='k',
               markersize=7, label='Group mean'),
        Line2D([0], [0], marker='o', lw=0, color='0.4', markersize=6,
               alpha=0.6, label='Individual subject'),
    ]
    fig.legend(handles=handles, ncol=2, loc='lower center', frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))

    label = '3-channel' if montage == '3ch' else 'Full montage'
    fig.suptitle(f'Per-subject cross-subject AUROC — {label}', fontsize=12,
                 y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    out = OUT_DIR / out_name
    fig.savefig(out, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  wrote {out}')


# ====================================================================
# Paired advantage heatmap (Table 2 companion)
# ====================================================================
# Per dataset x baseline: Hodges-Lehmann Delta of the paired per-subject AUROC
# differences (ERP-XTTN - baseline), colored by effect size; bold + boxed where
# the two-sided Wilcoxon signed-rank test is BH-significant (q<0.05) within each
# baseline family across the 9 datasets. Rows sorted easiest->hardest by mean AUROC
# across the 5 methods; dagger marks n<=6 (below the exact-test power floor).
# Recomputed from per-subject results.json via _persubj_auroc — no external inputs.

_HM_BASELINES = [('EEGNet', 'eegnet'), ('EEG-Deformer', 'eeg_deformer'),
                 ('EPMN', 'epmn'), ('xDAWN+RG', 'xdawn_rg')]
_HM_BLX = ['EEGNet', 'EEG-\nDeformer', 'EPMN', 'xDAWN\n+RG']


def _qsignrank(p, k):
    """p-quantile of the Wilcoxon signed-rank W+ null over k nonzero diffs (exact)."""
    M = k * (k + 1) // 2
    dp = [0] * (M + 1); dp[0] = 1
    for r in range(1, k + 1):
        for s in range(M, r - 1, -1):
            dp[s] += dp[s - r]
    tot = 2 ** k; c = 0
    for x in range(len(dp)):
        c += dp[x]
        if c / tot >= p:
            return x
    return M


def _hl_ci(diff, alpha=0.05):
    """Hodges-Lehmann estimate (median of Walsh averages) and its exact
    distribution-free signed-rank confidence interval, in the units of ``diff``.
    The CI excludes 0 iff the two-sided signed-rank test rejects at ``alpha`` — so
    the figure's boxing and the table's CI can never visually contradict."""
    dd = np.asarray([x for x in diff if x != 0.0]); k = dd.size
    if k == 0:
        return np.nan, np.nan, np.nan
    w = np.sort(np.array([(dd[i] + dd[j]) / 2 for i in range(k) for j in range(i, k)]))
    M = w.size; qu = max(_qsignrank(alpha / 2, k), 1); ql = M - qu
    return float(np.median(w)), float(w[qu - 1]), float(w[ql])


def _bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (monotone); NaNs passed through."""
    p = np.asarray(pvals, float); out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]; m = idx.size
    if m == 0:
        return out
    order = idx[np.argsort(p[idx])]; prev = 1.0; adj = np.empty(m)
    for rank in range(m - 1, -1, -1):
        prev = min(prev, p[order[rank]] * m / (rank + 1)); adj[rank] = prev
    for rank, i in enumerate(order):
        out[i] = min(adj[rank], 1.0)
    return out


def _paired_stats():
    """Single source of truth for the paired figure AND table (so they never drift).

    Per dataset x baseline (3-channel): Hodges-Lehmann Delta of the paired per-subject
    AUROC differences (ERP-XTTN - baseline) with its exact signed-rank 95% CI, the
    two-sided Wilcoxon raw p, and the BH-adjusted q across ALL 36 tests. Rows are
    difficulty-sorted (easiest = highest mean AUROC across the 5 methods, at top).
    Returns HL/LO/HI in raw AUROC units.
    """
    METH = ['erpxttn_peak'] + [m for _, m in _HM_BASELINES]
    au, mean_auroc = {}, {}
    for name, ds_dir, c3, cf, k3, kf, det in DATASETS:
        au[ds_dir] = {mm: _persubj_auroc(ds_dir, c3, mm) for mm in METH}
        mean_auroc[ds_dir] = float(np.mean([
            np.mean(list(au[ds_dir][mm].values())) for mm in METH]))
    order = sorted(DATASETS, key=lambda r: -mean_auroc[r[1]])
    labels = [d[0].replace(' ErrP', '') for d in order]
    n, m = len(order), len(_HM_BASELINES)
    HL = np.full((n, m), np.nan); LO = np.full((n, m), np.nan); HI = np.full((n, m), np.nan)
    praw = np.full((n, m), np.nan); N = np.zeros(n, int)
    for i, row in enumerate(order):
        xt = au[row[1]]['erpxttn_peak']
        for j, (_, bl) in enumerate(_HM_BASELINES):
            b = au[row[1]][bl]
            subs = sorted(set(xt) & set(b))
            if not subs:
                continue
            diff = np.array([xt[s] for s in subs]) - np.array([b[s] for s in subs])
            N[i] = len(subs)
            HL[i, j], LO[i, j], HI[i, j] = _hl_ci(diff)
            try:
                praw[i, j] = wilcoxon(diff, alternative='two-sided').pvalue
            except ValueError:
                praw[i, j] = np.nan
    # BH across ALL 36 tests (global FDR control); not inflated when signal
    # concentrates in one baseline column, unlike per-baseline families.
    Q = _bh_fdr(praw.ravel()).reshape(praw.shape)
    return dict(order=order, labels=labels, N=N, HL=HL, LO=LO, HI=HI,
                praw=praw, Q=Q, mean_auroc=mean_auroc)


def fig_paired_heatmap(out_path):
    """9x4 annotated heatmap of paired ERP-XTTN vs baseline effects (3-channel)."""
    st = _paired_stats()
    labels, N, Q = st['labels'], st['N'], st['Q']
    M = st['HL'] * 100.0            # Hodges-Lehmann Delta in AUROC points
    n, m = M.shape
    colmean = np.nanmean(M, axis=0)

    from matplotlib import cm, colors
    from matplotlib.patches import Rectangle
    v = np.nanmax(np.abs(M)); norm = colors.Normalize(-v, v)
    sm = cm.ScalarMappable(norm=norm, cmap='RdBu_r')

    fig, ax = plt.subplots(figsize=(6.6, 7.6))
    ax.imshow(M, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    for i in range(n):
        for j in range(m):
            sig = np.isfinite(Q[i, j]) and Q[i, j] < 0.05
            ax.text(j, i, f'{M[i, j]:+.1f}', ha='center', va='center', fontsize=10,
                    fontweight='bold' if sig else 'normal')
            if sig:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='k', lw=2.1))

    GAP = 0.55; my = n - 1 + GAP + 1
    for j in range(m):          # bottom margin: mean Delta per baseline (descriptive)
        ax.add_patch(Rectangle((j - .5, my - .5), 1, 1, facecolor=sm.to_rgba(colmean[j]),
                               ec='0.4', lw=.6))
        ax.text(j, my, f'{colmean[j]:+.1f}', ha='center', va='center', fontsize=9.5, style='italic')
    ax.text(-0.9, my, r'mean $\Delta$', ha='right', va='center', fontsize=8, style='italic', color='0.3')

    # difficulty arrow (rows are sorted by mean AUROC; easiest at top)
    ax.annotate('', xy=(-0.82, n - 0.6), xytext=(-0.82, -0.4),
                arrowprops=dict(arrowstyle='-|>', color='0.65', lw=1.2))
    ax.text(-1.12, n / 2 - 0.5, r'harder $\leftarrow$ easier', rotation=90,
            ha='center', va='center', fontsize=7.5, color='0.5')

    ax.set_xticks(range(m)); ax.set_xticklabels(_HM_BLX, fontsize=9.5)
    ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'{l} ($n{{=}}{N[i]}$)' + ('$^\\dagger$' if N[i] <= 6 else '')
                        for i, l in enumerate(labels)], fontsize=9)
    ax.set_xlim(-1.45, m - 0.4); ax.set_ylim(my + 0.7, -1.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle('Paired advantage of ERP-XTTN vs baselines (3-channel)\n'
                 'Hodges–Lehmann $\\Delta$ (AUROC points, ERP-XTTN $-$ baseline)', fontsize=11, y=0.99)

    cb = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.03, shrink=0.65)
    cb.set_ticks([-5, -2.5, 0, 2.5, 5])
    cb.set_ticklabels([f'{t:+.1f}'.replace('+0.0', '0') for t in [-5, -2.5, 0, 2.5, 5]])
    cb.ax.tick_params(labelsize=8)
    cb.set_label('HL $\\Delta$ (AUROC pts): red = ERP-XTTN higher, blue = lower', fontsize=8)
    fig.text(0.5, 0.058, 'Bold text + black box: $q<0.05$ (Wilcoxon signed-rank, Benjamini–Hochberg',
             ha='center', fontsize=8)
    fig.text(0.5, 0.030, 'FDR across all 36 tests).   $^\\dagger$ $n\\leq6$: below exact-test power floor.',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=0.17, right=0.92, top=0.855, bottom=0.11)
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(Path(out_path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def write_paired_table(out_path):
    """LaTeX companion table for Table 2 — SAME stats as fig_paired_heatmap via
    _paired_stats() (Hodges-Lehmann Delta, exact signed-rank 95% CI, global-36 BH).
    Cell = HL Delta over its CI; bold = q<0.05. Needs \\usepackage{booktabs,makecell,amsmath}."""
    st = _paired_stats()
    labels, N, HL, LO, HI, Q = st['labels'], st['N'], st['HL'], st['LO'], st['HI'], st['Q']
    n, m = HL.shape
    L = [r'\begin{table}[t]', r'\centering',
         (r'\caption{Paired comparison of ERP-XTTN against each baseline (3-channel montage), '
          r'companion to Table~\ref{tab:table2}. Each cell reports the Hodges--Lehmann estimate of the '
          r'paired per-subject AUROC difference $\Delta=\mathrm{AUROC}_{\text{ERP-XTTN}}-\mathrm{AUROC}_{\text{baseline}}$ '
          r'(negative $=$ ERP-XTTN lower) with its 95\% exact distribution-free signed-rank CI below. '
          r'Per-subject AUROCs are averaged over 5 seeds first (xDAWN+RG is deterministic). \textbf{Bold} $=$ '
          r'two-sided Wilcoxon signed-rank significant at $q<0.05$ after Benjamini--Hochberg FDR correction across '
          r'all 36 comparisons (9 datasets $\times$ 4 baselines). $^\dagger$ On BNCI ($n=6$) the exact test floors '
          r'at $p=0.031$ and cannot survive FDR correction regardless of effect size; BNCI is therefore reported '
          r'descriptively, as the Hodges--Lehmann point estimate with the CI omitted.}'),
         r'\label{tab:paired}', r'\small', r'\setlength{\tabcolsep}{5pt}',
         r'\renewcommand{\arraystretch}{1.1}', r'\begin{tabular}{lcccc}', r'\toprule',
         r'Dataset & vs EEGNet & vs EEG-Deformer & vs EPMN & vs xDAWN+RG \\',
         r'        & \multicolumn{4}{c}{\footnotesize Hodges--Lehmann $\Delta$ (95\% CI)} \\',
         r'\midrule']
    for i in range(n):
        lab = f'{labels[i]} ($n{{=}}{N[i]}$)' + (r'$^\dagger$' if N[i] <= 6 else '')
        cells = []
        for j in range(m):
            if N[i] <= 6:       # descriptive: point estimate only, CI omitted at n<=6
                body = f'${HL[i, j]:+.3f}$'
            else:
                body = f'${HL[i, j]:+.3f}$\\\\ \\footnotesize[${LO[i, j]:+.3f}$,\\,${HI[i, j]:+.3f}$]'
            cell = r'\makecell[c]{%s}' % body
            if np.isfinite(Q[i, j]) and Q[i, j] < 0.05:
                cell = r'{\boldmath\makecell[c]{%s}}' % body
            cells.append(cell)
        L.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
        if i < n - 1:
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    Path(out_path).write_text('\n'.join(L) + '\n')


# ====================================================================
# Main
# ====================================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    print('Per-subject AUROC (individual points)...')
    fig_persubject_auroc('3ch', 'fig_persubject_auroc_3ch.png')
    fig_persubject_auroc('full', 'fig_persubject_auroc_full.png')

    print('Tax-driver scatter...')
    fig_tax_drivers()

    print('SNR-vs-gap scatter (main-text)...')
    fig_snr_gap()

    print('TP↔FP / TP↔TN bar chart...')
    fig_tpfp_tptn()

    print('Paired advantage heatmap (Table 2 companion)...')
    fig_paired_heatmap(OUT_DIR / 'fig_paired_heatmap.png')

    print('Paired advantage table (Table 2 companion, LaTeX)...')
    write_paired_table(OUT_DIR / 'table_paired.tex')

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
