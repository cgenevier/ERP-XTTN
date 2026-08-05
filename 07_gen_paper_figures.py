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
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
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


def _persubj_auroc_msd(ds_dir, var_dir, model):
    """{subject: (mean, sd)} of LOSO AUROC across seeds."""
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
    return {s: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for s, v in acc.items()}


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
    ('ERN',       'erpcore_ern',         'noref_midline3_ern_rs256_iir_fwd_bp-1-10',     'noref_rs256_iir_fwd_bp-1-10', 'erpcore_ern|midline3_ern',     'erpcore_ern|full',         0),  # FCz, Cz, Pz
    ('LRP',       'erpcore_lrp',         'noref_lateral3_lrp_rs256_iir_fwd_bp-1-10',     'noref_rs256_iir_fwd_bp-1-10', 'erpcore_lrp|lateral3_lrp',     'erpcore_lrp|full',         0),  # C3, Cz, C4 — det=C3
    ('HRI ErrP',  'hri_errp_cursor',     'noref_midline3_iir_fwd_bp-1-10',               'noref_iir_fwd_bp-1-10',       'hri_errp_cursor|midline3',     'hri_errp_cursor|full',     1),
    ('BNCI ErrP', 'bnci_errp_013-2015',  'noref_midline3_rs256_iir_fwd_bp-1-10',         'noref_rs256_iir_fwd_bp-1-10', 'bnci_errp_013-2015|midline3',  'bnci_errp_013-2015|full',  1),
    ('N170',      'erpcore_n170',        'noref_occipital3_n170_rs256_iir_fwd_bp-1-10',  'noref_rs256_iir_fwd_bp-1-10', 'erpcore_n170|occipital3_n170', 'erpcore_n170|full',        2),  # P7, Oz, P8 — det=P7
    ('P300',      'erpcore_p300',        'noref_midline3_rs256_iir_fwd_bp-1-10',         'noref_rs256_iir_fwd_bp-1-10', 'erpcore_p300|midline3',        'erpcore_p300|full',        2),
    ('N2pc',      'erpcore_n2pc',        'noref_posterior3_n2pc_rs256_iir_fwd_bp-1-10',  'noref_rs256_iir_fwd_bp-1-10', 'erpcore_n2pc|posterior3_n2pc', 'erpcore_n2pc|full',        0),  # PO7, Pz, PO8 — det=PO7
    ('MMN',       'erpcore_mmn',         'noref_midline3_rs256_iir_fwd_bp-1-10',         'noref_rs256_iir_fwd_bp-1-10', 'erpcore_mmn|midline3',         'erpcore_mmn|full',         0),
    ('N400',      'erpcore_n400',        'noref_midline3_n400_rs256_iir_fwd_bp-1-10',    'noref_rs256_iir_fwd_bp-1-10', 'erpcore_n400|midline3_n400',   'erpcore_n400|full',        1),  # Cz, CPz, Pz — det=Cz idx 0
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
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
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
    fig.suptitle('Interpretability Gap vs Signal Strength', fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = OUT_DIR / 'fig_snr_gap.png'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
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
    ax.set_title('TP↔FP and TP↔TN Waveform Correlations Across Datasets',
                 fontsize=11, pad=10)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT_DIR / 'fig_tpfp_tptn.png'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'  wrote {out}')


# ====================================================================
# C. Peak-unit routing figures
# ====================================================================


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



def _load_subject_epochs(epoch_root, subj, pos_key, neg_key, label_groups):
    """Load one subject's epochs with event filtering, returning (X_uV, y, ch_names).

    Returns (None, None, None) if no matching epochs are found.
    Shared by the morphology cache builder and the per-seed correlation analysis."""
    import mne
    subj_dir = epoch_root / subj
    fifs = sorted(subj_dir.rglob('*-epo.fif'))
    if not fifs:
        return None, None, None
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
    if not all_X:
        return None, None, None
    return np.concatenate(all_X, axis=0) * 1e6, np.concatenate(all_y, axis=0), ch_names


def _build_morphology_cache(ds_label, ds_dir, var_dir, det_idx, cache_path):
    """Compute per-subject TP/FN/TN/FP grand-mean waveforms at the
    detection channel from epoched_fif + predictions, save to npz.

    Uses the same event filtering as 06_gen_analysis.py / training:
    keep epochs whose event name matches pos_key or neg_key (or
    label_groups if defined), drop everything else."""

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
        pf = rdir / f'two_factor_{subj}.npz'
        pred = np.load(pf if pf.exists() else rdir / f'predictions_{subj}.npz')
        probs = pred['probs']
        labels = pred['labels']

        X_uV, y_loaded, ch_names = _load_subject_epochs(
            epoch_root, subj, pos_key, neg_key, label_groups)
        if X_uV is None:
            print(f'    {subj}: no matching events, skipping')
            continue

        if ch_names_all is None:
            ch_names_all = ch_names

        if X_uV.shape[0] != len(labels):
            print(f'    {subj}: epoch count mismatch ({X_uV.shape[0]} vs '
                  f'predictions {len(labels)}), skipping')
            continue
        if not np.array_equal(y_loaded, labels):
            print(f'    {subj}: label order mismatch, skipping')
            continue

        if n_times is None:
            n_times = X_uV.shape[2]

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)

    for ax_idx, (a_m, a_s, a_label, a_color, b_m, b_s, b_label, b_color, panel_name) in enumerate([
        (tp_m, tp_s, f'TP (hit, n={n_tp} subj)', '#197b30',
         fn_m, fn_s, f'FN (miss, n={n_fn} subj)', '#b22222', 'TP / FN'),
        (tn_m, tn_s, f'TN (correct reject, n={n_tn} subj)', '#1f4e79',
         fp_m, fp_s, f'FP (false alarm, n={n_fp} subj)', '#d97706', 'TN / FP'),
    ]):
        ax = axes[ax_idx]
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
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_title(panel_name, fontsize=9.5, pad=2)
        ax.legend(loc='best', fontsize=7.5, frameon=False)
        ax.tick_params(axis='both', labelsize=8)

    axes[0].set_ylabel('Amplitude (µV)', fontsize=9)

    fig.tight_layout(h_pad=0.5)
    top = max(ax.get_position().y1 for ax in axes)
    fig.text(0.5, top + 0.06, f'{ds_label} — Outcome-Conditioned Grand Averages at {det_name} (3-Channel)',
             fontsize=10, ha='center', va='bottom')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
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

    datasets_ordered = _difficulty_order()

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

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
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


def _proto_fold_names(proto_raw, proto_windows, det_idx, sfreq):
    """Polarity-derived prototype names (P1, P2, ... / N1, N2, ...)."""
    K = proto_raw.shape[0]
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
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.7, 1.0, 1.0], hspace=0.32, wspace=0.18)
    axp = fig.add_subplot(gs[0, :])
    for k in range(len(names)):
        c = PEAK_PROTO_COLORS[k % len(PEAK_PROTO_COLORS)]
        axp.axvspan(windows[k][0], windows[k][1], color=c, alpha=0.16)
        s, e = int(windows[k][0] / 1000 * sfreq), int(windows[k][1] / 1000 * sfreq)
        axp.plot(time_ms[s:e], proto_det[k, s:e], color=c, lw=2.5,
                 label=f'{names[k]} ({windows[k][0]:.0f}-{windows[k][1]:.0f} ms)')
    axp.axhline(0, color='gray', lw=0.5, ls='--'); axp.set_xlim(0, time_ms[-1])
    axp.set_title('Difference-Wave Prototypes', fontsize=13, fontweight='bold', pad=2)
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
    top = axp.get_position().y1
    fig.text(0.5, top + 0.02, f'Peak-Unit Routing — {ds_label} ({subjects[0]} vs {subjects[1]})',
             fontsize=15, fontweight='bold', ha='center', va='bottom')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.02); plt.close(fig)
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
    rdir_protos = sorted(rdir.glob('prototypes_*.npz'))
    mean_proto = np.zeros((K, 3, T))
    for f in rdir_protos:
        d = np.load(f)
        p = d['mf_template']
        mean_proto[:p.shape[0]] += p
    mean_proto /= max(len(rdir_protos), 1)
    proto_names = _proto_fold_names(mean_proto, windows, det_idx, sfreq)
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
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
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
        C_full = np.load(proto_files[0])['mf_template'].shape[1]
        mean_proto = np.zeros((K, C_full, T))
        for f in proto_files:
            p = np.load(f)['mf_template']
            mean_proto[:p.shape[0]] += p
        mean_proto /= max(len(proto_files), 1)
        per_ds.append({
            'name': name, 'subjects': subjects, 'protos': all_protos,
            'per_fold_windows': per_fold_windows, 'fold_K': fold_K,
            'K': K, 'windows': windows, 'time_ms': time_ms,
            'proto_names': _proto_fold_names(mean_proto, windows, det_idx, sfreq),
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
            ds['name'], xy=(-0.62, 0.5), xycoords='axes fraction',
            ha='center', va='center', fontsize=10, fontweight='bold',
            rotation=90,
        )

    fig.suptitle(
        'Difference-Wave Prototypes (LOSO Folds) Across Datasets — '
        '3-Channel, Detection Channel Only',
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0.06, 0, 1, 0.98))
    out = OUT_DIR / 'fig_prototypes_all_datasets.png'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
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

    for di, row in enumerate(_difficulty_order()):
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

    label = '3-Channel' if montage == '3ch' else 'Full Montage'
    fig.suptitle(f'LOSO AUROC by Subject — {label}', fontsize=12,
                 y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    out = OUT_DIR / out_name
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
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
    RRB = np.full((n, m), np.nan)
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
            nz = diff[diff != 0]
            if nz.size > 0:
                ranks = np.argsort(np.argsort(np.abs(nz))) + 1
                Wplus = float(np.sum(ranks[nz > 0]))
                Ntot = nz.size * (nz.size + 1) / 2
                RRB[i, j] = 2 * Wplus / Ntot - 1
            try:
                praw[i, j] = wilcoxon(diff, alternative='two-sided').pvalue
            except ValueError:
                praw[i, j] = np.nan
    # BH across ALL 36 tests (global FDR control); not inflated when signal
    # concentrates in one baseline column, unlike per-baseline families.
    Q = _bh_fdr(praw.ravel()).reshape(praw.shape)
    return dict(order=order, labels=labels, N=N, HL=HL, LO=LO, HI=HI,
                praw=praw, Q=Q, RRB=RRB, mean_auroc=mean_auroc)


def fig_paired_heatmap(out_path):
    """4x9 horizontal annotated heatmap of paired ERP-XTTN vs baseline effects
    (3-channel). Baselines on rows, datasets on columns (difficulty order)."""
    st = _paired_stats()
    labels, N, Q = st['labels'], st['N'], st['Q']
    M = st['HL'] * 100.0
    n, m = M.shape  # n=9 datasets, m=4 baselines

    Mt = M.T   # (4, 9) — baselines × datasets
    Qt = Q.T
    rowmean = np.nanmean(Mt, axis=1)

    from matplotlib import cm, colors
    from matplotlib.patches import Rectangle
    v = np.nanmax(np.abs(Mt)); norm = colors.Normalize(-v, v)
    sm = cm.ScalarMappable(norm=norm, cmap='RdBu_r')

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.imshow(Mt, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    for i in range(m):
        for j in range(n):
            sig = np.isfinite(Qt[i, j]) and Qt[i, j] < 0.05
            ax.text(j, i, f'{Mt[i, j]:+.1f}', ha='center', va='center', fontsize=10,
                    fontweight='bold' if sig else 'normal')
            if sig:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='k', lw=2.1))

    GAP = 0.55; mx = n - 1 + GAP + 1
    for i in range(m):
        ax.add_patch(Rectangle((mx - .5, i - .5), 1, 1, facecolor=sm.to_rgba(rowmean[i]),
                               ec='0.4', lw=.6))
        ax.text(mx, i, f'{rowmean[i]:+.1f}', ha='center', va='center', fontsize=9.5, style='italic')
    ax.text(mx, -0.9, 'mean', ha='center', va='bottom', fontsize=8, style='italic', color='0.3')

    ax.annotate('', xy=(n - 0.6, -0.82), xytext=(-0.4, -0.82),
                arrowprops=dict(arrowstyle='-|>', color='0.65', lw=1.2))
    ax.text(n / 2 - 0.5, -1.12, r'easier $\rightarrow$ harder',
            ha='center', va='center', fontsize=7.5, color='0.5')

    ds_labels = [f'{l}\n($n{{=}}{N[j]}$)' + ('$^\\dagger$' if N[j] <= 6 else '')
                 for j, l in enumerate(labels)]
    ax.set_xticks(range(n)); ax.set_xticklabels(ds_labels, fontsize=8.5)
    ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
    ax.set_yticks(range(m)); ax.set_yticklabels(_HM_BLX, fontsize=9.5)
    ax.set_ylim(m - 0.5, -1.4); ax.set_xlim(-0.5, mx + 0.7)
    ax.tick_params(length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.suptitle('Paired Advantage of ERP-XTTN vs Baselines (3-Channel)',
                 fontsize=11, y=0.98)

    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.8)
    cb.set_ticks([-5, -2.5, 0, 2.5, 5])
    cb.set_ticklabels([f'{t:+.1f}'.replace('+0.0', '0') for t in [-5, -2.5, 0, 2.5, 5]])
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Hodges–Lehmann $\\Delta$ (AUROC pts)', fontsize=8)
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.10, right=0.92)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(Path(out_path).with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.02)
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
          r'paired per-subject AUROC difference (ERP-XTTN minus baseline; negative $=$ ERP-XTTN lower) '
          r'with its 95\% exact distribution-free signed-rank CI below. '
          r'Per-subject AUROCs are averaged over 5 seeds first (xDAWN+RG is deterministic). \textbf{Bold} $=$ '
          r'two-sided Wilcoxon signed-rank significant at $q<0.05$ after Benjamini--Hochberg FDR correction across '
          r'all 36 comparisons (9 datasets $\times$ 4 baselines). $^\dagger$ On BNCI ($n=6$) the exact test floors '
          r'at $p=0.031$ and cannot survive FDR correction regardless of effect size; BNCI is therefore reported '
          r'descriptively, as the Hodges--Lehmann point estimate with the CI omitted.}'),
         r'\label{tab:paired}', r'\setlength{\tabcolsep}{5pt}',
         r'\renewcommand{\arraystretch}{1.1}', r'\begin{tabular}{lcccc}', r'\toprule',
         r'Dataset & vs EEGNet & vs EEG-Deformer & vs EPMN & vs xDAWN+RG \\',
         r'        & \multicolumn{4}{c}{\footnotesize Hodges--Lehmann paired difference (95\% CI)} \\',
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
# Supplementary LaTeX tables
# ====================================================================
# All share the same data helpers (results.json / validation.json / morphology
# cache) so they stay consistent with the figures. booktabs required throughout.
_SUP_METHODS = [('ERP-XTTN', 'erpxttn_peak'), ('EEGNet', 'eegnet'),
                ('EEG-Deformer', 'eeg_deformer'), ('EPMN', 'epmn'), ('xDAWN+RG', 'xdawn_rg')]
def _difficulty_order():
    """Datasets sorted by mean AUROC across all 5 methods, descending (hardest last).
    Same order as the paired heatmap / paired table."""
    METH = [m for _, m in _SUP_METHODS]
    mean_auroc = {}
    for d in DATASETS:
        ds_dir, v3 = d[1], d[2]
        vals = []
        for mdl in METH:
            pj = _persubj_fusion(ds_dir, v3, mdl) if mdl == 'erpxttn_peak' else _persubj_forward(ds_dir, v3, mdl)
            m = _msd(pj)[0]
            if m is not None:
                vals.append(m)
        mean_auroc[ds_dir] = float(np.mean(vals)) if vals else 0
    return sorted(DATASETS, key=lambda r: -mean_auroc[r[1]])


def _sup_rows():
    """Datasets in difficulty order (highest mean AUROC first), matching Table 2."""
    return [(d[0].replace(' ErrP', ''), d) for d in _difficulty_order()]


def _persubj_fusion(ds_dir, var_dir, model):
    """{subj: seed-avg two-factor AUROC} — the fused headline."""
    from collections import defaultdict
    base = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / model
    acc = defaultdict(list)
    for sd in sorted(base.glob('seed-*')):
        rj = sd / 'results.json'
        if not rj.exists():
            continue
        for s, a in (json.load(open(rj)).get('two_factor_auroc_per_subject') or {}).items():
            if a is not None:
                acc[s].append(float(a))
    return {s: float(np.mean(v)) for s, v in acc.items() if v}


def _persubj_forward(ds_dir, var_dir, model):
    """{subj: seed-avg per-fold test AUROC} — the model's own forward output."""
    from collections import defaultdict
    base = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / model
    acc = defaultdict(list)
    for sd in sorted(base.glob('seed-*')):
        rj = sd / 'results.json'
        if not rj.exists():
            continue
        for f in json.load(open(rj)).get('folds', []):
            if f.get('test_auroc') is not None:
                acc[f['test_subject']].append(float(f['test_auroc']))
    return {s: float(np.mean(v)) for s, v in acc.items() if v}


def _persubj_balacc(ds_dir, var_dir, model):
    """{subj: seed-avg balanced accuracy @0.5}. ERP-XTTN uses the fused decision."""
    from collections import defaultdict
    from sklearn.metrics import balanced_accuracy_score
    base = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / model
    acc = defaultdict(list)
    for sd in sorted(base.glob('seed-*')):
        if model == 'erpxttn_peak':
            for f in sd.glob('two_factor_*.npz'):
                s = f.name[len('two_factor_'):-4]
                d = np.load(f)
                y = d['labels'].astype(int)
                if len(np.unique(y)) < 2:
                    continue
                acc[s].append(balanced_accuracy_score(y, (d['probs'] >= 0.5).astype(int)))
        else:
            rj = sd / 'results.json'
            if not rj.exists():
                continue
            for f in json.load(open(rj)).get('folds', []):
                if f.get('test_bal_acc') is not None:
                    acc[f['test_subject']].append(float(f['test_bal_acc']))
    return {s: float(np.mean(v)) for s, v in acc.items() if v}


def _msd(d):
    v = list(d.values())
    if not v:
        return (None, None, 0)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))


def _backfill_from_folds(vj, agg):
    """Add aggregate keys that were computed per-fold but never aggregated."""
    _BACKFILL = {
        'ladder_phase': 'swap_ladder.phase',
        'ladder_reversed': 'swap_ladder.reversed',
    }
    folds = vj.get('folds', [])
    if not folds:
        return
    for agg_key, fold_path in _BACKFILL.items():
        if agg_key in agg:
            continue
        parts = fold_path.split('.')
        vals = []
        for f in folds:
            v = f
            for p in parts:
                v = v.get(p, {}) if isinstance(v, dict) else {}
            if isinstance(v, (int, float)):
                vals.append(v)
        if vals:
            agg[agg_key] = {'mean': float(np.mean(vals)),
                            'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            'n_folds': len(vals)}


def _val_agg_sup(ds_dir, var_dir):
    """Seed-averaged validation.json aggregate; (dict_mean, dict_sd, n_seeds).
    dict_mean: {key: seed-averaged mean across subjects}.
    dict_sd:   {key: seed-averaged SD across subjects}."""
    base = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / 'erpxttn_peak'
    acc_mean = {}
    acc_sd = {}
    ns = 0
    for s in range(1, 6):
        p = base / f'seed-{s}' / 'validation.json'
        if not p.exists():
            continue
        vj = json.load(open(p))
        agg = vj.get('aggregate', {})
        if not agg:
            continue
        _backfill_from_folds(vj, agg)
        ns += 1
        for k, v in agg.items():
            if isinstance(v, dict) and 'mean' in v:
                acc_mean.setdefault(k, []).append(v['mean'])
                if 'sd' in v:
                    acc_sd.setdefault(k, []).append(v['sd'])
    if not acc_mean:
        return (None, None, 0)
    return ({k: float(np.mean(v)) for k, v in acc_mean.items()},
            {k: float(np.mean(v)) for k, v in acc_sd.items()},
            ns)


def _tex(out_path, lines):
    Path(out_path).write_text('\n'.join(lines) + '\n')


# --- S3: full-montage AUROC ---
def write_fullmontage_auroc_table(out_path):
    L = [r'% Table S1 — Full-montage AUROC',
         r'\begin{table}[t]', r'\centering',
         r'\setlength{\tabcolsep}{5pt}',
         r'\caption{%',
         r'Full-montage classification performance (area under the receiver operating '
         r'characteristic curve, AUROC; mean $\pm$ standard deviation across '
         r'subjects) under leave-one-subject-out '
         r'(LOSO) cross-validation. Channel counts vary by dataset (Table~\ref{tab:datasets} of '
         r'the main text): 64 for BNCI, 27 for HRI, and 30 for all ERP CORE '
         r'datasets. Neural models (ERP-XTTN, EEGNet, EEG-Deformer, EPMN) '
         r'report seed-averaged means across five independent training seeds; '
         r'xDAWN+RG is deterministic (single run). Bold marks the highest mean '
         r'per dataset. $\Delta$ = best baseline minus ERP-XTTN '
         r'(performance gap, in AUROC); the best baseline is selected '
         r'post hoc per dataset and $\Delta$ is reported descriptively. '
         r'Datasets ordered as in main text Table~\ref{tab:auroc} (three-channel mean AUROC, descending). '
         r'AUROC: area under the receiver operating characteristic curve; '
         r'LOSO: leave-one-subject-out; EEG-Deformer: dense convolutional '
         r'transformer \cite{ding2024}; EPMN: ERP Prototypical Matching Net '
         r'\cite{wei2022}; xDAWN+RG: xDAWN spatial filtering with Riemannian '
         r'geometry classification.%',
         r'}',
         r'\label{sup:tab-fullmontage-auroc}',
         r'\begin{tabular}{lcccccc}', r'\toprule',
         r'Dataset & ERP-XTTN & EEGNet & EEG-Deformer & EPMN & xDAWN+RG & $\Delta$ \\', r'\midrule']
    all_rounded = {n: [] for n, _ in _SUP_METHODS}
    all_deltas = []
    for lab, d in _sup_rows():
        vf = d[3]
        vals = {}
        for name, mdl in _SUP_METHODS:
            pj = _persubj_fusion(d[1], vf, mdl) if mdl == 'erpxttn_peak' else _persubj_forward(d[1], vf, mdl)
            m, s, n = _msd(pj)
            vals[name] = (round(m, 3) if m is not None else None,
                          round(s, 3) if s is not None else None)
        best = max((v[0] for v in vals.values() if v[0] is not None), default=None)
        cells = []
        for name, _ in _SUP_METHODS:
            m, s = vals[name]
            if m is None:
                cells.append('--')
            else:
                all_rounded[name].append(m)
                cell = f'{m:.3f} $\\pm$ {s:.3f}'
                if best is not None and abs(m - best) < 1e-4:
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
        xt_m = vals['ERP-XTTN'][0]
        bl = [vals[n][0] for n, _ in _SUP_METHODS if n != 'ERP-XTTN' and vals[n][0] is not None]
        if bl and xt_m is not None:
            delta = round(max(bl) - xt_m, 3)
            all_deltas.append(delta)
            L.append(f'{lab} & ' + ' & '.join(cells) + f' & \\textit{{{delta:.3f}}} \\\\')
        else:
            L.append(f'{lab} & ' + ' & '.join(cells) + r' & -- \\')
    L.append(r'\midrule')
    mean_cells = []
    for name, _ in _SUP_METHODS:
        v = all_rounded[name]
        m = round(float(np.mean(v)), 3) if v else None
        if m is None:
            mean_cells.append('--')
        else:
            mean_cells.append(f'{m:.3f}')
    best_mean = max(round(float(np.mean(all_rounded[n])), 3) for n, _ in _SUP_METHODS if all_rounded[n])
    for i, (name, _) in enumerate(_SUP_METHODS):
        v = all_rounded[name]
        if v and abs(round(float(np.mean(v)), 3) - best_mean) < 1e-4:
            mean_cells[i] = r'\textbf{' + mean_cells[i] + '}'
    mean_delta = round(float(np.mean(all_deltas)), 3) if all_deltas else None
    delta_cell = f'\\textbf{{{mean_delta:.3f}}}' if mean_delta is not None else '--'
    L.append(r'\textit{Mean} & ' + ' & '.join(mean_cells) + f' & {delta_cell} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


def write_fullmontage_ba_table(out_path):
    L = [r'% Table S2 — Full-montage balanced accuracy',
         r'\begin{table}[t]', r'\centering',
         r'\setlength{\tabcolsep}{5pt}',
         r'\caption{%',
         r'Full-montage balanced accuracy (mean $\pm$ standard deviation across subjects, '
         r'LOSO) at threshold 0.5. The decision '
         r'threshold was fixed at 0.5 on the predicted positive-class '
         r'probability and applied uniformly across all subjects and '
         r'configurations, simulating deployment conditions where per-subject '
         r'threshold optimization is unavailable. Bold marks the highest mean '
         r'per dataset. $\Delta$ = best baseline minus ERP-XTTN. ERP-XTTN '
         r'reports the fused (routing $+$ amplitude) decision; neural models '
         r'report seed-averaged means across five training seeds; xDAWN+RG is '
         r'deterministic. Datasets ordered as in main text Table~\ref{tab:auroc} '
         r'(three-channel mean AUROC, descending). Three-channel balanced accuracy '
         r'is in Table~\ref{sup:tab-balacc}. '
         r'Abbreviations as in Table~\ref{sup:tab-fullmontage-auroc}.%',
         r'}',
         r'\label{sup:tab-fullmontage-ba}',
         r'\begin{tabular}{lcccccc}', r'\toprule',
         r'Dataset & ERP-XTTN & EEGNet & EEG-Deformer & EPMN & xDAWN+RG & $\Delta$ \\', r'\midrule']
    all_rounded = {n: [] for n, _ in _SUP_METHODS}
    all_deltas = []
    for lab, d in _sup_rows():
        vf = d[3]
        vals = {}
        for name, mdl in _SUP_METHODS:
            pj = _persubj_balacc(d[1], vf, mdl)
            m, s, n = _msd(pj)
            vals[name] = (round(m, 3) if m is not None else None,
                          round(s, 3) if s is not None else None)
        best = max((v[0] for v in vals.values() if v[0] is not None), default=None)
        cells = []
        for name, _ in _SUP_METHODS:
            m, s = vals[name]
            if m is None:
                cells.append('--')
            else:
                all_rounded[name].append(m)
                cell = f'{m:.3f} $\\pm$ {s:.3f}'
                if best is not None and abs(m - best) < 1e-4:
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
        xt_m = vals['ERP-XTTN'][0]
        bl = [vals[n][0] for n, _ in _SUP_METHODS if n != 'ERP-XTTN' and vals[n][0] is not None]
        if bl and xt_m is not None:
            delta = round(max(bl) - xt_m, 3)
            all_deltas.append(delta)
            L.append(f'{lab} & ' + ' & '.join(cells) + f' & \\textit{{{delta:.3f}}} \\\\')
        else:
            L.append(f'{lab} & ' + ' & '.join(cells) + r' & -- \\')
    L.append(r'\midrule')
    mean_cells = []
    for name, _ in _SUP_METHODS:
        v = all_rounded[name]
        m = round(float(np.mean(v)), 3) if v else None
        mean_cells.append(f'{m:.3f}' if m is not None else '--')
    best_mean = max(round(float(np.mean(all_rounded[n])), 3) for n, _ in _SUP_METHODS if all_rounded[n])
    for i, (name, _) in enumerate(_SUP_METHODS):
        v = all_rounded[name]
        if v and abs(round(float(np.mean(v)), 3) - best_mean) < 1e-4:
            mean_cells[i] = r'\textbf{' + mean_cells[i] + '}'
    mean_delta = round(float(np.mean(all_deltas)), 3) if all_deltas else None
    delta_cell = f'\\textbf{{{mean_delta:.3f}}}' if mean_delta is not None else '--'
    L.append(r'\textit{Mean} & ' + ' & '.join(mean_cells) + f' & {delta_cell} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


def write_balacc_table(out_path):
    L = [r'% Table S6 — 3-channel balanced accuracy',
         r'\begin{table}[t]', r'\centering',
         r'\setlength{\tabcolsep}{5pt}',
         r'\caption{%',
         r'Balanced accuracy (mean $\pm$ standard deviation across subjects, '
         r'LOSO) at threshold 0.5, three-channel montage. Conventions and '
         r'abbreviations as in Table~\ref{sup:tab-fullmontage-ba}. Bold marks the highest mean per '
         r'dataset. $\Delta$ = best baseline minus ERP-XTTN. Full-montage '
         r'balanced accuracy is in Table~\ref{sup:tab-fullmontage-ba}; AUROC values at the '
         r'three-channel montage are in main text Table~\ref{tab:auroc}.%',
         r'}',
         r'\label{sup:tab-balacc}',
         r'\begin{tabular}{lcccccc}', r'\toprule',
         r'Dataset & ERP-XTTN & EEGNet & EEG-Deformer & EPMN & xDAWN+RG & $\Delta$ \\', r'\midrule']
    all_rounded = {n: [] for n, _ in _SUP_METHODS}
    all_deltas = []
    for lab, d in _sup_rows():
        v3 = d[2]
        vals = {}
        for name, mdl in _SUP_METHODS:
            pj = _persubj_balacc(d[1], v3, mdl)
            m, s, n = _msd(pj)
            vals[name] = (round(m, 3) if m is not None else None,
                          round(s, 3) if s is not None else None)
        best = max((v[0] for v in vals.values() if v[0] is not None), default=None)
        cells = []
        for name, _ in _SUP_METHODS:
            m, s = vals[name]
            if m is None:
                cells.append('--')
            else:
                all_rounded[name].append(m)
                cell = f'{m:.3f} $\\pm$ {s:.3f}'
                if best is not None and abs(m - best) < 1e-4:
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
        xt_m = vals['ERP-XTTN'][0]
        bl = [vals[n][0] for n, _ in _SUP_METHODS if n != 'ERP-XTTN' and vals[n][0] is not None]
        if bl and xt_m is not None:
            delta = round(max(bl) - xt_m, 3)
            all_deltas.append(delta)
            L.append(f'{lab} & ' + ' & '.join(cells) + f' & \\textit{{{delta:.3f}}} \\\\')
        else:
            L.append(f'{lab} & ' + ' & '.join(cells) + r' & -- \\')
    L.append(r'\midrule')
    mean_cells = []
    for name, _ in _SUP_METHODS:
        v = all_rounded[name]
        m = round(float(np.mean(v)), 3) if v else None
        mean_cells.append(f'{m:.3f}' if m is not None else '--')
    best_mean = max(round(float(np.mean(all_rounded[n])), 3) for n, _ in _SUP_METHODS if all_rounded[n])
    for i, (name, _) in enumerate(_SUP_METHODS):
        v = all_rounded[name]
        if v and abs(round(float(np.mean(v)), 3) - best_mean) < 1e-4:
            mean_cells[i] = r'\textbf{' + mean_cells[i] + '}'
    mean_delta = round(float(np.mean(all_deltas)), 3) if all_deltas else None
    delta_cell = f'\\textbf{{{mean_delta:.3f}}}' if mean_delta is not None else '--'
    L.append(r'\textit{Mean} & ' + ' & '.join(mean_cells) + f' & {delta_cell} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


_PERSUBJ_DS_INFO = {
    'HRI': ('HRI ErrP', 11),
    'ERN': ('ERP CORE ERN', 40),
    'LRP': ('ERP CORE LRP', 40),
    'BNCI': ('BNCI ErrP', 6),
    'N170': ('ERP CORE N170', 40),
    'P300': ('ERP CORE P300', 40),
    'N2pc': ('ERP CORE N2pc', 40),
    'MMN': ('ERP CORE MMN', 40),
    'N400': ('ERP CORE N400', 40),
}


def write_persubject_auroc_tables(out_path):
    parts = []
    for ti, (lab, d) in enumerate(_sup_rows()):
        v3 = d[2]
        pm = {name: _persubj_auroc_msd(d[1], v3, mdl) for name, mdl in _SUP_METHODS}
        subs = sorted(set().union(*[set(x) for x in pm.values() if x])) if any(pm.values()) else []
        full_name, n_subj = _PERSUBJ_DS_INFO.get(lab, (lab, '?'))
        if ti == 0:
            cap = (
                r'\caption{%' '\n'
                r'Per-subject AUROC (mean $\pm$ standard deviation across five '
                r'training seeds, LOSO) at the three-channel montage for '
                + full_name + r' ($n = ' + str(n_subj) + r'$ subjects). '
                r'Subject numbering follows the original dataset release, which does '
                r'not include sub-01 or sub-12; all eleven subjects distributed in '
                r'the public release were analyzed. '
                r'Each row is one LOSO fold with the indicated '
                r'subject held out; the reported value is that subject\textquotesingle s test-set '
                r'AUROC averaged across seeds (xDAWN+RG: single deterministic run, '
                r'standard deviation omitted). Bold marks the highest mean per '
                r'subject. $\Delta$ = best baseline minus ERP-XTTN. The mean row '
                r'matches the ' + lab + r' entry of main text Table~\ref{tab:auroc}. '
                r'$\Delta$ is computed per subject as that subject\textquotesingle s best-baseline '
                r'AUROC minus ERP-XTTN AUROC, then averaged; because the identity of '
                r'the best baseline varies across subjects, the mean $\Delta$ here is '
                r'generally larger than the fixed-best-baseline $\Delta$ reported in '
                r'Table~\ref{tab:auroc}. AUROC: area under '
                r'the receiver operating characteristic curve; LOSO: '
                r'leave-one-subject-out.%' '\n'
                r'}')
        else:
            cap = (
                r'\caption{%' '\n'
                r'Per-subject AUROC (mean $\pm$ standard deviation across five '
                r'training seeds, LOSO) at the three-channel montage for '
                + full_name + r' ($n = ' + str(n_subj) + r'$ subjects). '
                r'Conventions as in Table~\ref{sup:tab-persubj-hri}. The mean '
                r'row matches the ' + lab + r' entry of main text Table~\ref{tab:auroc}.%' '\n'
                r'}')
        P = [f'% Table S{7 + ti} — Per-subject AUROC, {lab}',
             r'\begin{table}[t]', r'\centering',
             cap,
             r'\label{sup:tab-persubj-%s}' % lab.lower().replace(' ', ''),
             r'\begin{tabular}{lcccccc}', r'\toprule',
             r'Subject & ERP-XTTN & EEGNet & EEG-Deformer & EPMN & xDAWN+RG & $\Delta$ \\', r'\midrule']
        all_means = {n: [] for n, _ in _SUP_METHODS}
        all_deltas = []
        for s in subs:
            row_vals = {}
            for name, _ in _SUP_METHODS:
                row_vals[name] = pm[name].get(s) if pm[name] else None
            best = max((v[0] for v in row_vals.values() if v is not None), default=None)
            cells = []
            for name, _ in _SUP_METHODS:
                v = row_vals[name]
                if v is None:
                    cells.append('--')
                else:
                    m, sd = v
                    all_means[name].append(m)
                    cell = f'{m:.3f} $\\pm$ {sd:.3f}' if sd > 0 else f'{m:.3f}'
                    if best is not None and abs(m - best) < 1e-4:
                        cell = r'\textbf{' + cell + '}'
                    cells.append(cell)
            xt = row_vals['ERP-XTTN']
            bl = [row_vals[n][0] for n, _ in _SUP_METHODS if n != 'ERP-XTTN' and row_vals[n] is not None]
            if bl and xt is not None:
                delta = max(bl) - xt[0]
                all_deltas.append(delta)
                P.append(f'{s} & ' + ' & '.join(cells) + f' & \\textit{{{delta:.3f}}} \\\\')
            else:
                P.append(f'{s} & ' + ' & '.join(cells) + r' & -- \\')
        P.append(r'\midrule')
        mean_cells = []
        for name, _ in _SUP_METHODS:
            v = all_means[name]
            if v:
                m, s_val = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                mean_cells.append(f'{m:.3f} $\\pm$ {s_val:.3f}')
            else:
                mean_cells.append('--')
        means = {n: float(np.mean(all_means[n])) for n, _ in _SUP_METHODS if all_means[n]}
        if means:
            best_mean = max(means.values())
            for i, (name, _) in enumerate(_SUP_METHODS):
                if name in means and abs(means[name] - best_mean) < 1e-4:
                    mean_cells[i] = r'\textbf{' + mean_cells[i] + '}'
        mean_delta = float(np.mean(all_deltas)) if all_deltas else None
        delta_cell = f'\\textbf{{{mean_delta:.3f}}}' if mean_delta is not None else '--'
        P.append(r'\textit{Mean} & ' + ' & '.join(mean_cells) + f' & {delta_cell} \\\\')
        P += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
        parts.append('\n'.join(P))
    _tex(out_path, ['% Per-subject AUROC tables, one per dataset (S7--S15).', ''] + ['\n\n'.join(parts)])


# --- S6: ablation grid ---
_ABL_ARMS = [('Base (routing+amplitude)', 'erpxttn_peak', 'fusion'),
             ('Routing only', 'erpxttn_peak', 'forward'),
             ('Amplitude only', 'erpxttn_peak_amp_only', 'fusion'),
             ('End-to-end head', 'ablation_erpxttn_e2e', 'forward'),
             ('No whitening', 'ablation_erpxttn_nowhiten', 'fusion'),
             ('No self-attention', 'erpxttn_peak_nosa', 'fusion'),
             ('$K=2$', 'erpxttn_peak_k2', 'fusion'), ('$K=6$', 'erpxttn_peak_k6', 'fusion'),
             ('Prominence 0.01', 'erpxttn_peak_prom0.01', 'fusion'),
             ('Prominence 0.05', 'erpxttn_peak_prom0.05', 'fusion'),
             ('2 heads', 'erpxttn_peak_h2', 'fusion'), ('8 heads', 'erpxttn_peak_h8', 'fusion'),
             ('Learned free head', 'ablation_erpxttn_learned_readout', 'forward')]
_ABL_DS = [('HRI', 'hri_errp_cursor'), ('ERN', 'erpcore_ern'),
           ('P300', 'erpcore_p300'), ('N400', 'erpcore_n400')]


def write_ablation_table(out_path):
    by = {d[1]: d for d in DATASETS}
    L = [r'% Table S21 — Two-factor ablation grid',
         r'\begin{table}[t]', r'\centering', r'\setlength{\tabcolsep}{6pt}',
         r'\caption{%',
         r'Architecture ablation grid at the three-channel montage '
         r'(seed-averaged across five seeds). Each ablation is a single-knob '
         r'deviation from the default ERP-XTTN configuration of Section~\ref{sec:architecture}, '
         r'evaluated on four datasets spanning the difficulty range (HRI, '
         r'ERN, P300, N400) under the five-seed LOSO protocol. AUROC is '
         r'reported as the mean $\pm$ standard deviation across subjects. The '
         r'base configuration uses both routing and amplitude factors '
         r'combined by $L_2$-penalized logistic regression on frozen '
         r'per-trial outputs (Section~\ref{sec:amplitude}). Ablation arms: routing only '
         r'and amplitude only remove one factor from the combiner; end-to-end '
         r'trains the routing pathway and the combiner jointly by '
         r'backpropagation rather than fitting the combiner on frozen '
         r'outputs; no whitening replaces the whitened cosine '
         r'(Equation~3) with a raw cosine; no self-attention removes the '
         r'self-attention layer from the peak-embedding pathway; $K = 2$ '
         r'and $K = 6$ vary the maximum prototype count against the default '
         r'$K = 4$; prominence varies the peak-detection threshold '
         r'(default $\pi = 0.02$); 2 heads and 8 heads vary the '
         r'attention-head count (default $H = 4$); learned free head '
         r'replaces the grounded readout with the unconstrained learned '
         r'classification head of the original ERP-XTTN \cite{wyman2026}. '
         r'$\Delta$ = ablation arm AUROC minus base AUROC (negative '
         r'indicates the ablation reduced performance). AUROC: area under '
         r'the receiver operating characteristic curve; LOSO: '
         r'leave-one-subject-out.%',
         r'}',
         r'\label{sup:tab-ablation}', r'\begin{tabular}{llcc}', r'\toprule',
         r'Dataset & Ablation & AUROC (mean $\pm$ SD) & $\Delta$ vs base \\', r'\midrule']
    for di, (lab, dsdir) in enumerate(_ABL_DS):
        v3 = by[dsdir][2]
        basem = _msd(_persubj_fusion(dsdir, v3, 'erpxttn_peak'))[0]
        for ai, (arm, mdir, metric) in enumerate(_ABL_ARMS):
            pj = _persubj_fusion(dsdir, v3, mdir) if metric == 'fusion' else _persubj_forward(dsdir, v3, mdir)
            m, sd, n = _msd(pj)
            dcell = lab if ai == 0 else ''
            if m is None:
                L.append(f'{dcell} & {arm} & -- & -- ' + r'\\')
                continue
            dl = '--' if arm.startswith('Base') or basem is None else f'{m - basem:+.3f}'
            L.append(f'{dcell} & {arm} & {m:.3f} $\\pm$ {sd:.3f} & {dl} ' + r'\\')
        if di < len(_ABL_DS) - 1:
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


def _learned_head_agg(ds_dir, var_dir):
    """Seed-averaged learned-head ladder_forward from validation_learned.json.
    Returns (dict_mean, dict_sd) or (None, None)."""
    base = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / 'ablation_erpxttn_learned_readout'
    acc_mean = {}
    acc_sd = {}
    ns = 0
    for s in range(1, 6):
        p = base / f'seed-{s}' / 'validation_learned.json'
        if not p.exists():
            continue
        lf = json.load(open(p)).get('ladder_forward', {})
        if not lf:
            continue
        ns += 1
        for k, v in lf.items():
            if isinstance(v, dict) and 'mean' in v:
                acc_mean.setdefault(k, []).append(v['mean'])
                if 'sd' in v:
                    acc_sd.setdefault(k, []).append(v['sd'])
    if not acc_mean:
        return None, None
    return ({k: float(np.mean(v)) for k, v in acc_mean.items()},
            {k: float(np.mean(v)) for k, v in acc_sd.items()})


def write_polarity_contrast_table(out_path):
    by = {d[1]: d for d in DATASETS}
    L = [r'% Table S22 — Polarity-inversion contrast: grounded vs learned readout',
         r'\begin{table}[t]', r'\centering', r'\setlength{\tabcolsep}{6pt}',
         r'\caption{%',
         r'Effect of prototype polarity inversion on the grounded routing readout '
         r'(routing factor only) versus the unconstrained learned classification head of the original '
         r'ERP-XTTN \cite{wyman2026}, at the three-channel montage '
         r'(mean AUROC, LOSO, seed-averaged across five seeds). '
         r'Both readouts operate over the identical attention tensor and identical '
         r'frozen prototypes; only the readout differs. Inversion is applied at '
         r'inference on the frozen model. Noise-null values for the grounded arm are '
         r'in Table~\ref{sup:tab-grounding-3-routing}. '
         r'Datasets ordered by mean AUROC across all five methods, descending. '
         r'Abbreviations as in Table~\ref{sup:tab-grounding-full-routing}.%',
         r'}',
         r'\label{sup:tab-polarity-contrast}',
         r'\begin{tabular}{lcccc}', r'\toprule',
         r'Dataset & ERP-XTTN Routing-only & Polarity-inv. '
         r'& Learned Head (Intact) & Learned Head (Polarity-inv.) \\',
         r'\midrule']
    def _msd(m, sd):
        if m is None:
            return '--'
        return f'{m:.2f} $\\pm$ {sd:.2f}' if sd is not None else f'{m:.2f}'
    for lab, dsdir in _ABL_DS:
        v3 = by[dsdir][2]
        gagg, gagg_sd, _ = _val_agg_sup(dsdir, v3)
        gi = gagg.get('routing_auroc') if gagg else None
        gi_sd = gagg_sd.get('routing_auroc') if gagg_sd else None
        gp = gagg.get('ladder_polarity') if gagg else None
        gp_sd = gagg_sd.get('ladder_polarity') if gagg_sd else None
        lagg, lagg_sd = _learned_head_agg(dsdir, v3)
        li = lagg.get('intact') if lagg else None
        li_sd = lagg_sd.get('intact') if lagg_sd else None
        lp = lagg.get('polarity') if lagg else None
        lp_sd = lagg_sd.get('polarity') if lagg_sd else None
        cells = [_msd(gi, gi_sd), _msd(gp, gp_sd), _msd(li, li_sd), _msd(lp, lp_sd)]
        L.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


def write_wilcoxon_table(out_path):
    st = _paired_stats()
    labels, N = st['labels'], st['N']
    HL, LO, HI = st['HL'], st['LO'], st['HI']
    praw, Q, RRB = st['praw'], st['Q'], st['RRB']
    n, m = HL.shape
    bl_names = [h for h, _ in _HM_BASELINES]
    L = [r'% Table S16 — Wilcoxon signed-rank test statistics',
         r'\begin{table}[t]', r'\centering', r'\setlength{\tabcolsep}{4pt}',
         r'\caption{%',
         r'Paired Wilcoxon signed-rank tests for ERP-XTTN versus each baseline '
         r'at the three-channel montage (companion to main text Figure~\ref{fig:fig_paired_heatmap}). '
         r'Per-subject AUROCs are averaged over five training seeds '
         r'before pairing (xDAWN+RG: single deterministic run). '
         r'$\hat{\Delta}$: Hodges-Lehmann estimate of the paired difference '
         r'(ERP-XTTN minus baseline; negative = ERP-XTTN lower). '
         r'95\% CI: exact distribution-free signed-rank confidence interval. '
         r'$p$: two-sided Wilcoxon signed-rank raw $p$-value. '
         r'$q$: Benjamini-Hochberg adjusted $p$-value controlling the '
         r'false-discovery rate across all 36 comparisons '
         r'(9 datasets $\times$ 4 baselines). '
         r'$r_{rb}$: rank-biserial correlation (matched-pairs), '
         r'computed as $r_{rb} = 2W^{+} / [n(n+1)/2] - 1$ where $W^{+}$ is '
         r'the sum of positive-difference ranks; values near $+1$/$-1$ '
         r'indicate ERP-XTTN consistently higher/lower. '
         r'Bold $q$ marks comparisons significant at $q < 0.05$. '
         r'$^\dagger$BNCI ($n=6$): the exact signed-rank test floors at '
         r'$p = 0.031$ and cannot survive FDR correction regardless of effect '
         r'size; results are reported descriptively (CI omitted). '
         r'Datasets ordered by mean AUROC across all five methods, descending. '
         r'AUROC: area under the receiver operating characteristic curve; '
         r'LOSO: leave-one-subject-out.%',
         r'}',
         r'\label{sup:tab-wilcoxon}',
         r'\begin{tabular}{llccccc}', r'\toprule',
         r'Dataset & Baseline & $\hat{\Delta}$ & 95\% CI & $p$ & $q$ & $r_{rb}$ \\',
         r'\midrule']
    for i in range(n):
        dagger = r'$^\dagger$' if N[i] <= 6 else ''
        ds_lab = f'{labels[i]} ($n{{=}}{N[i]}$){dagger}'
        for j in range(m):
            dcell = ds_lab if j == 0 else ''
            hl_s = f'${HL[i, j]:+.3f}$'
            if N[i] <= 6:
                ci_s = '--'
            else:
                ci_s = f'[${LO[i, j]:+.3f}$, ${HI[i, j]:+.3f}$]'
            if np.isfinite(praw[i, j]):
                if praw[i, j] < 0.001:
                    p_s = '$<$0.001'
                else:
                    p_s = f'{praw[i, j]:.3f}'
            else:
                p_s = '--'
            if np.isfinite(Q[i, j]):
                q_sig = Q[i, j] < 0.05
                if Q[i, j] < 0.001:
                    q_s = '$<$0.001'
                else:
                    q_s = f'{Q[i, j]:.3f}'
                if q_sig:
                    q_s = r'\textbf{' + q_s + '}'
            else:
                q_s = '--'
            if np.isfinite(RRB[i, j]):
                r_s = f'${RRB[i, j]:+.3f}$'
            else:
                r_s = '--'
            L.append(f'{dcell} & {bl_names[j]} & {hl_s} & {ci_s} & {p_s} & {q_s} & {r_s} ' + r'\\')
        if i < n - 1:
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


# --- S7 / S8: grounding interventions ---
_GND_ROUTING_COLS = [
    ('two_factor', 'ERP-XTTN'),
    ('routing_auroc', 'Routing-only'),
    ('ladder_reversed', 'Time-rev.'),
    ('ladder_null', 'Noise null'),
    ('ladder_phase', 'Phase-rand.'),
    ('ladder_polarity', 'Polarity-inv.'),
    ('ladder_cross', 'Cross-comp.'),
    ('carrier_peak_proto', 'Proto-perm'),
    ('carrier_trial', 'Trial-perm'),
]

_GND_AMP_COLS = [
    ('two_factor', 'ERP-XTTN'),
    ('amp_channel_on', 'On-target'),
    ('amp_channel_baseline', 'Off-target (early)'),
    ('amp_channel_off', 'Off-target (late)'),
    ('amp_channel_permute', 'Permutation'),
    ('amp_contrast_on', 'On-target'),
    ('amp_contrast_baseline', 'Off-target (early)'),
    ('amp_contrast_off', 'Off-target (late)'),
    ('amp_contrast_permute', 'Permutation'),
]


def _grounding_body(cols, montage, snum, label_suffix, comment, caption_lines):
    vi = 2 if montage == '3-channel' else 3
    L = [f'% Table {snum} — {comment}',
         r'\begin{table}[t]', r'\centering',
         r'\setlength{\tabcolsep}{5pt}']
    L += caption_lines
    L.append(r'\label{sup:tab-grounding-%s-%s}' % (montage.split('-')[0], label_suffix))
    col_accum = {k: [] for k, _ in cols}
    data_rows = []
    for lab, d in _sup_rows():
        agg, _, _ = _val_agg_sup(d[1], d[vi])
        if not agg:
            data_rows.append((lab, None))
            continue
        data_rows.append((lab, agg))
        for k, _ in cols:
            v = agg.get(k)
            if v is not None:
                col_accum[k].append(v)
    return L, col_accum, data_rows, vi


def write_grounding_routing_table(out_path, montage, snum):
    if montage == 'full-montage':
        cap = [
            r'\caption{%',
            r'Full-montage routing grounding interventions '
            r'(mean AUROC, LOSO, seed-averaged across five seeds). The ERP-XTTN '
            r'column reports the full two-factor model AUROC (from '
            r'Table~\ref{sup:tab-fullmontage-auroc}) as '
            r'a reference. All remaining columns report the routing-factor AUROC '
            r'under the indicated intervention, applied to each fold\textquotesingle s frozen '
            r'trained model without retraining. Routing-only: unmodified prototypes '
            r'(intact reference). Time-rev: prototype waveforms time-reversed. '
            r'Noise null: prototypes replaced '
            r'by variance-matched Gaussian noise, establishing the baseline '
            r'discriminability recoverable from routing-pattern geometry alone. '
            r'Phase-rand: prototype waveforms phase-randomized (amplitude spectrum '
            r'preserved, phase shuffled). '
            r'Polarity-inv: prototypes negated. Cross-comp: '
            r'match-pathway states are cyclically shifted across prototype slots '
            r'while routing keys remain fixed. '
            r'Proto-perm: prototype columns of the match matrix are permuted '
            r'independently within each trial. Trial-perm: which trial supplies each match value is '
            r'permuted. Datasets ordered by mean AUROC across all five methods, descending. '
            r'Standard deviations are omitted for compactness; fold-level and '
            r'seed-level variability are available in the released code '
            r'\cite{wyman2026erpxttn_code}. AUROC: area under the receiver operating '
            r'characteristic curve; LOSO: leave-one-subject-out.%',
            r'}']
    else:
        cap = [
            r'\caption{%',
            r'Routing grounding interventions at the three-channel montage '
            r'(mean AUROC, LOSO, seed-averaged across five seeds). The ERP-XTTN '
            r'column reports the full two-factor model AUROC (from main text '
            r'Table~\ref{tab:auroc}) as a reference. Column definitions are identical to '
            r'Table~\ref{sup:tab-grounding-full-routing}. '
            r'Datasets ordered by mean AUROC across all five methods, descending. '
            r'Standard deviations are omitted and are available in the released code '
            r'\cite{wyman2026erpxttn_code}. '
            r'Abbreviations as in Table~\ref{sup:tab-grounding-full-routing}.%',
            r'}']
    L, col_accum, data_rows, _ = _grounding_body(
        _GND_ROUTING_COLS, montage, snum, 'routing',
        f'Routing grounding, {montage}', cap)
    hdr = 'Dataset & ' + ' & '.join(h for _, h in _GND_ROUTING_COLS) + r' \\'
    ncols = len(_GND_ROUTING_COLS)
    L += [r'\begin{tabular}{l' + 'c' * ncols + '}', r'\toprule', hdr, r'\midrule']
    for lab, agg in data_rows:
        if agg is None:
            L.append(f'{lab} & ' + ' & '.join(['--'] * ncols) + r' \\')
            continue
        cells = []
        for k, _ in _GND_ROUTING_COLS:
            m = agg.get(k)
            cells.append(f'{m:.2f}' if m is not None else '--')
        L.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
    L.append(r'\midrule')
    mean_cells = []
    for k, _ in _GND_ROUTING_COLS:
        v = col_accum[k]
        mean_cells.append(f'{float(np.mean(v)):.2f}' if v else '--')
    L.append(r'\textit{Mean} & ' + ' & '.join(mean_cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


def write_grounding_amp_table(out_path, montage, snum):
    if montage == 'full-montage':
        cap = [
            r'\caption{%',
            r'Full-montage amplitude grounding controls '
            r'(mean AUROC, LOSO, seed-averaged across five seeds). The ERP-XTTN '
            r'column reports the full two-factor model AUROC (from '
            r'Table~\ref{sup:tab-fullmontage-auroc}) as '
            r'a reference. Amplitude features are evaluated separately for the '
            r'monopolar channel matched filter (Channel MF; Equation~5) and the '
            r'bipolar contrast matched filter (Contrast MF; Equation~6). For '
            r'each derivation: On-target = matched filter computed over the '
            r"prototype's detected component window; Off-target (early) = window "
            r'displaced $-300$\,ms from the detected latency (clipped to epoch '
            r'start); Off-target (late) = window displaced $+150$\,ms; Permutation = '
            r'matched-filter features shuffled across trials. For each fold and feature '
            r'family, AUROC is the maximum orientation-invariant AUROC over '
            r'individual features. These values constitute a descriptive '
            r'best-feature diagnostic rather than the performance of a trained '
            r'classifier or the logistic combiner. Datasets ordered '
            r'by mean AUROC across all five methods, descending. Standard deviations '
            r'are omitted and are available in the released code \cite{wyman2026erpxttn_code}. '
            r'Abbreviations as in Table~\ref{sup:tab-grounding-full-routing}.%',
            r'}']
    else:
        cap = [
            r'\caption{%',
            r'Amplitude grounding controls at the three-channel montage '
            r'(mean AUROC, LOSO, seed-averaged across five seeds). The ERP-XTTN '
            r'column reports the full two-factor model AUROC from main text '
            r'Table~\ref{tab:auroc} as a reference. Column definitions are identical to '
            r'Table~\ref{sup:tab-grounding-full-amp}. For each fold and feature '
            r'family, AUROC is the maximum orientation-invariant AUROC over '
            r'individual features. These values constitute a descriptive '
            r'best-feature diagnostic rather than the performance of a trained '
            r'classifier or the logistic combiner. At three channels the '
            r'diagnostic searches across $KC$ monopolar features and '
            r'$K\binom{C}{2}$ bipolar features, for up to 24 features in total '
            r'at $K=4$ and $C=3$. Datasets ordered by '
            r'mean AUROC across all five methods, descending. Standard deviations '
            r'are omitted and are available in the released code '
            r'\cite{wyman2026erpxttn_code}. Abbreviations as in '
            r'Table~\ref{sup:tab-grounding-full-routing}.%',
            r'}']
    L, col_accum, data_rows, _ = _grounding_body(
        _GND_AMP_COLS, montage, snum, 'amp',
        f'Amplitude grounding, {montage}', cap)
    ncols = len(_GND_AMP_COLS)
    L += [r'\begin{tabular}{l c cccc cccc}', r'\toprule',
          r' & & \multicolumn{4}{c}{Channel MF} & \multicolumn{4}{c}{Contrast MF} \\',
          r'\cmidrule(lr){3-6} \cmidrule(lr){7-10}']
    hdr = 'Dataset & ' + ' & '.join(h for _, h in _GND_AMP_COLS) + r' \\'
    L += [hdr, r'\midrule']
    for lab, agg in data_rows:
        if agg is None:
            L.append(f'{lab} & ' + ' & '.join(['--'] * ncols) + r' \\')
            continue
        cells = []
        for k, _ in _GND_AMP_COLS:
            m = agg.get(k)
            cells.append(f'{m:.2f}' if m is not None else '--')
        L.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
    L.append(r'\midrule')
    mean_cells = []
    for k, _ in _GND_AMP_COLS:
        v = col_accum[k]
        mean_cells.append(f'{float(np.mean(v)):.2f}' if v else '--')
    L.append(r'\textit{Mean} & ' + ' & '.join(mean_cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


# --- S9: EPMN native recipe ---
def write_epmn_native_table(out_path):
    by = {d[1]: d for d in DATASETS}
    rows = [('ERN', 'erpcore_ern'), ('P300', 'erpcore_p300'), ('N400', 'erpcore_n400')]
    L = [r'% Table S5 — EPMN native vs shared protocol',
         r'\begin{table}[t]', r'\centering', r'\setlength{\tabcolsep}{6pt}',
         r'\caption{%',
         r'EPMN native-recipe versus shared-protocol comparison (mean $\pm$ '
         r'standard deviation across subjects, LOSO) at the three-channel '
         r'montage. The native recipe follows the training procedure of '
         r'Wei et al.\ \cite{wei2022}: input resampled to 64 time points; '
         r'SGD optimizer (learning rate $10^{-4}$, momentum 0.9) with '
         r'learning-rate halving every 5 epochs; episodic training with '
         r'$N_q = 12$ query trials and $N_s = 18$ support trials per class, '
         r're-sampled each episode; majority-class down-sampling with '
         r'unweighted cross-entropy; no augmentation. The shared protocol '
         r'uses the AdamW optimizer, cosine-annealed learning rate, temporal '
         r'jitter, and additive noise described in Section~\ref{sec:training} of the main '
         r'text. Both protocols use the same two-phase subject-level '
         r'validation scaffolding and LOSO folds. Bold marks the higher mean '
         r'per dataset. $\Delta$ = shared-protocol minus native AUROC '
         r'(positive indicates the shared protocol is higher). The '
         r'comparison was run on the three datasets for which the native '
         r'EPMN episodic recipe could be implemented from the original '
         r'description. EPMN: ERP Prototypical Matching Net; '
         r'abbreviations otherwise as in Table~\ref{sup:tab-fullmontage-auroc}.%',
         r'}',
         r'\label{sup:tab-epmn-native}',
         r'\begin{tabular}{lccc}', r'\toprule',
         r'Dataset & Native AUROC & Shared-protocol AUROC & $\Delta$ \\', r'\midrule']
    all_nat, all_sh, all_deltas = [], [], []
    for lab, dsdir in rows:
        v3 = by[dsdir][2]
        nat_m, nat_s, _ = _msd(_persubj_forward(dsdir, v3, 'epmn_native'))
        sh_m, sh_s, _ = _msd(_persubj_forward(dsdir, v3, 'epmn'))
        nat_cell = f'{nat_m:.3f} $\\pm$ {nat_s:.3f}' if nat_m is not None else '--'
        sh_cell = f'{sh_m:.3f} $\\pm$ {sh_s:.3f}' if sh_m is not None else '--'
        if nat_m is not None and sh_m is not None:
            best = max(nat_m, sh_m)
            if abs(nat_m - best) < 1e-4:
                nat_cell = r'\textbf{' + nat_cell + '}'
            if abs(sh_m - best) < 1e-4:
                sh_cell = r'\textbf{' + sh_cell + '}'
            delta = sh_m - nat_m
            all_deltas.append(delta)
            all_nat.append(nat_m)
            all_sh.append(sh_m)
            L.append(f'{lab} & {nat_cell} & {sh_cell} & \\textit{{{delta:+.3f}}} \\\\')
        else:
            L.append(f'{lab} & {nat_cell} & {sh_cell} & -- \\\\')
    L.append(r'\midrule')
    mn = f'{float(np.mean(all_nat)):.3f}' if all_nat else '--'
    ms = f'{float(np.mean(all_sh)):.3f}' if all_sh else '--'
    if all_nat and all_sh:
        best_mean = max(float(np.mean(all_nat)), float(np.mean(all_sh)))
        if abs(float(np.mean(all_nat)) - best_mean) < 1e-4:
            mn = r'\textbf{' + mn + '}'
        if abs(float(np.mean(all_sh)) - best_mean) < 1e-4:
            ms = r'\textbf{' + ms + '}'
    md = f'\\textbf{{{float(np.mean(all_deltas)):+.3f}}}' if all_deltas else '--'
    L.append(f'\\textit{{Mean}} & {mn} & {ms} & {md} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


# --- S10: TP<->FP / TP<->TN correlations ---
def write_corr_table(out_path):
    from scipy.stats import pearsonr
    L = [r'% Table S17 — TP/FP and TP/TN waveform correlations',
         r'\begin{table}[t]', r'\centering', r'\setlength{\tabcolsep}{6pt}',
         r'\caption{%',
         r'Outcome-conditioned waveform correlations at the detection channel '
         r'(three-channel montage, fused decision). For each subject, '
         r'single-trial waveforms at the detection channel (Table~1) were '
         r'averaged separately for the four prediction$\times$label outcomes: '
         r'true positive (TP), false positive (FP), true negative (TN), and '
         r'false negative (FN). TP$\leftrightarrow$FP and '
         r'TP$\leftrightarrow$TN report the per-subject Pearson correlation '
         r'between the mean TP waveform and the mean FP (respectively TN) '
         r'waveform, summarized as mean $\pm$ standard deviation across '
         r'subjects. Values are computed from a single reference training seed '
         r'(seed 1); the standard deviations therefore reflect cross-subject '
         r'variability. Table S18 reports the corresponding cross-subject means '
         r'aggregated across all five seeds. '
         r'TP$\leftrightarrow$FP $>$ TP$\leftrightarrow$TN '
         r'indicates that false positives morphologically resemble true '
         r'positives more than true negatives do, consistent with '
         r'classification operating through waveform--prototype '
         r'correspondence (Section~3.2 of the main text). Det.\ ch: '
         r'detection channel on which prototypes are defined (Table~1).%',
         r'}',
         r'\label{sup:tab-corr}', r'\begin{tabular}{llcc}', r'\toprule',
         r'Dataset & Det.\ ch & TP$\leftrightarrow$FP & TP$\leftrightarrow$TN \\', r'\midrule']

    def _rs(A, B):
        rr = []
        for a, b in zip(A, B):
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            rr.append(pearsonr(a, b)[0])
        if not rr:
            return None, None
        return float(np.mean(rr)), (float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0)

    for lab, d in _sup_rows():
        ok = _ensure_morphology_cache(d[0], d[1], d[2], d[6])
        cache = CACHE_DIR / f'{d[1]}_3ch_Cz.npz'
        if not ok or not cache.exists():
            L.append(f'{lab} & -- & -- & -- ' + r'\\')
            continue
        z = np.load(cache, allow_pickle=True)
        det = str(z['det_channel_name'])
        fpm, fps = _rs(z['tp'], z['fp'])
        tnm, tns = _rs(z['tp'], z['tn'])
        fpc = f'${fpm:+.2f} \\pm {fps:.2f}$' if fpm is not None else '--'
        tnc = f'${tnm:+.2f} \\pm {tns:.2f}$' if tnm is not None else '--'
        L.append(f'{lab} & {det} & {fpc} & {tnc} ' + r'\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


def write_corr_perseed_table(out_path):
    """Table S11: per-seed stability of the TP<->FP > TP<->TN ordering.

    For each 3-channel dataset and each seed, computes the cross-subject mean
    Pearson r between detection-channel grand-mean TP and FP (and TN) waveforms
    using that seed's fused decision. Reports mean +/- SD across seeds."""
    from scipy.stats import pearsonr

    SEEDS = [1, 2, 3, 4, 5]

    def _r(a, b):
        if a is None or b is None:
            return np.nan
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return np.nan
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return float(pearsonr(a, b)[0])

    results = []
    for lab, d in _sup_rows():
        ds_dir, var_dir, det_idx = d[1], d[2], d[6]
        cfg = json.load(open(DATASETS_DIR / ds_dir / 'dataset_config.json'))
        pos_key = cfg['label_map']['pos_key']
        neg_key = cfg['label_map']['neg_key']
        label_groups = cfg.get('label_groups')
        det_name = cfg.get('detection_channel', '?')
        base = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / var_dir / 'erpxttn_peak'

        ref = base / 'seed-1' / 'results.json'
        if not ref.exists():
            print(f'  {lab}: no seed-1 results, skipping')
            continue
        subjects = [r['test_subject'] for r in json.load(open(ref))['folds']]

        epoch_root = DATASETS_DIR / ds_dir / 'epoched_fif' / 'tmin0ms_tmax800ms' / var_dir
        subj_cache = {}
        for subj in subjects:
            X_uV, y, _ = _load_subject_epochs(epoch_root, subj, pos_key, neg_key, label_groups)
            if X_uV is not None:
                subj_cache[subj] = (X_uV[:, det_idx, :], y)
        if not subj_cache:
            print(f'  {lab}: no epochs loaded, skipping')
            continue

        per_seed = []
        for seed in SEEDS:
            sdir = base / f'seed-{seed}'
            if not sdir.exists():
                continue
            fps, tns = [], []
            for subj, (det_sig, y) in subj_cache.items():
                src = sdir / f'two_factor_{subj}.npz'
                if not src.exists():
                    src = sdir / f'predictions_{subj}.npz'
                if not src.exists():
                    continue
                pred = np.load(src)
                probs, labels = pred['probs'], pred['labels']
                if len(labels) != len(y) or not np.array_equal(labels, y):
                    continue
                preds = (probs >= 0.5).astype(int)
                tp = (labels == 1) & (preds == 1)
                tn = (labels == 0) & (preds == 0)
                fp = (labels == 0) & (preds == 1)
                gm = lambda m: det_sig[m].mean(axis=0) if m.sum() > 0 else None
                tpm, tnm, fpm = gm(tp), gm(tn), gm(fp)
                rfp, rtn = _r(tpm, fpm), _r(tpm, tnm)
                if np.isfinite(rfp):
                    fps.append(rfp)
                if np.isfinite(rtn):
                    tns.append(rtn)
            if fps and tns:
                per_seed.append({'seed': seed, 'n_subj': len(fps),
                                 'tp_fp_mean': float(np.mean(fps)),
                                 'tp_tn_mean': float(np.mean(tns))})

        if not per_seed:
            print(f'  {lab}: no seeds computed, skipping')
            continue

        fp_means = np.array([s['tp_fp_mean'] for s in per_seed])
        tn_means = np.array([s['tp_tn_mean'] for s in per_seed])
        gaps = fp_means - tn_means
        rec = {
            'dataset': lab, 'det_channel': det_name, 'n_seeds': len(per_seed),
            'tp_fp_mean': float(fp_means.mean()), 'tp_fp_sd': float(fp_means.std()),
            'tp_tn_mean': float(tn_means.mean()), 'tp_tn_sd': float(tn_means.std()),
            'seeds_fp_gt_tn': int((gaps > 0).sum()),
            'min_margin': float(gaps.min()), 'max_margin': float(gaps.max()),
            'per_seed': per_seed,
        }
        results.append(rec)
        tag = 'OK' if rec['seeds_fp_gt_tn'] == rec['n_seeds'] else '*** FLIP ***'
        print(f'  {lab:5s}  TP-FP={rec["tp_fp_mean"]:+.3f}+/-{rec["tp_fp_sd"]:.3f}  '
              f'TP-TN={rec["tp_tn_mean"]:+.3f}+/-{rec["tp_tn_sd"]:.3f}  '
              f'FP>TN {rec["seeds_fp_gt_tn"]}/{rec["n_seeds"]}  [{tag}]')

    L = [r'% Table S18 — Per-seed waveform correlation stability',
         r'\begin{table}[t]', r'\centering',
         r'\setlength{\tabcolsep}{6pt}',
         r'\caption{%',
         r'Per-seed stability of the outcome-conditioned waveform ordering '
         r'(three-channel montage, fused decision), companion to '
         r'Table~\ref{sup:tab-corr}. Because Table S17 reports a single reference '
         r'seed, its values differ slightly from the five-seed means below. '
         r'For each dataset and training seed, '
         r'TP$\leftrightarrow$FP and TP$\leftrightarrow$TN are the '
         r'cross-subject mean of the per-subject Pearson correlations '
         r'defined in Table~\ref{sup:tab-corr}; columns report the mean $\pm$ standard '
         r'deviation of those cross-subject means across the five seeds. '
         r'FP${>}$TN seeds: number of seeds (out of five) on which '
         r'TP$\leftrightarrow$FP exceeded TP$\leftrightarrow$TN. Min.\ '
         r'margin: smallest per-seed difference '
         r'(TP$\leftrightarrow$FP $-$ TP$\leftrightarrow$TN). The ordering '
         r'TP$\leftrightarrow$FP $>$ TP$\leftrightarrow$TN held on all '
         r'five seeds for every dataset, indicating that the result is not '
         r'an artifact of a particular random initialization.%',
         r'}',
         r'\label{sup:tab-corr-perseed}',
         r'\begin{tabular}{llcccc}', r'\toprule',
         r'Dataset & Det.\ ch & TP$\leftrightarrow$FP & TP$\leftrightarrow$TN & '
         r'FP$>$TN seeds & Min.\ margin \\',
         r'\midrule']
    for r in results:
        L.append(
            f'{r["dataset"]} & {r["det_channel"]} & '
            f'${r["tp_fp_mean"]:+.2f} \\pm {r["tp_fp_sd"]:.2f}$ & '
            f'${r["tp_tn_mean"]:+.2f} \\pm {r["tp_tn_sd"]:.2f}$ & '
            f'{r["seeds_fp_gt_tn"]}/{r["n_seeds"]} & '
            f'${r["min_margin"]:+.2f}$ \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)

    json.dump(results, open(OUT_DIR / 's11_perseed_corr.json', 'w'), indent=2)
    n_ok = sum(1 for r in results if r['seeds_fp_gt_tn'] == r['n_seeds'])
    print(f'  Datasets with FP>TN in ALL seeds: {n_ok}/{len(results)}')
    flips = [r['dataset'] for r in results if r['seeds_fp_gt_tn'] != r['n_seeds']]
    if flips:
        print(f'  *** ordering NOT stable on: {", ".join(flips)} ***')


# --- Table 1: dataset summary ---
_EVENT_TYPE = {
    'bnci_errp_013-2015': 'Feedback', 'hri_errp_cursor': 'Feedback',
    'erpcore_ern': 'Response', 'erpcore_lrp': 'Response',
    'erpcore_mmn': 'Stimulus', 'erpcore_n170': 'Stimulus',
    'erpcore_n2pc': 'Stimulus', 'erpcore_n400': 'Stimulus',
    'erpcore_p300': 'Stimulus',
}
_T1_LABELS = {
    'BNCI ErrP': 'BNCI / ErrP', 'HRI ErrP': 'HRI / ErrP',
    'ERN': 'ERP CORE / ERN', 'LRP': 'ERP CORE / LRP',
    'MMN': 'ERP CORE / MMN', 'N170': 'ERP CORE / N170',
    'N2pc': 'ERP CORE / N2pc', 'N400': 'ERP CORE / N400',
    'P300': 'ERP CORE / P300',
}


# Row order for Table 1: the two ErrP datasets first, then ERP CORE by
# component (ERN, LRP, MMN, N170, N2pc, N400, P300). NOT difficulty order.
_T1_ORDER = [
    'bnci_errp_013-2015', 'hri_errp_cursor',
    'erpcore_ern', 'erpcore_lrp', 'erpcore_mmn',
    'erpcore_n170', 'erpcore_n2pc', 'erpcore_n400', 'erpcore_p300',
]


def write_dataset_table(out_path):
    L = [r'\begin{table}',
         r'\caption{Datasets evaluated. Electroencephalography (EEG) channels only; '
         r'electrooculography and reference channels were excluded where applicable. '
         r'Available Channels gives the full montage recorded for each dataset; the '
         r'3-channel montage is the evaluation condition reported in the main text, '
         r'chosen to cover the scalp region of the event-related potential (ERP) of '
         r'interest. The detection channel is the single electrode on which prototype '
         r'time windows are located by peak detection (Section~\ref{sec:architecture}); '
         r'for the ERP CORE datasets it is the canonical site reported for that '
         r'component by Kappenman et al.~\cite{kappenman2021}, and for BNCI and HRI it '
         r'is Cz, the central midline site available in both montages. Trials/Subject '
         r'refers to total classification epochs across both classes, summed across '
         r'sessions where applicable (BNCI includes two sessions per subject).}',
         r'\label{tab:datasets}',
         r'\centering',
         r'\begin{tabular}{lcccccc}', r'\hline',
         r'\textbf{Dataset / ERP} & \textbf{Time-Locked} & \textbf{Total} & '
         r'\textbf{Trials/Subject} & \textbf{Full} & \textbf{3-Ch} & '
         r'\textbf{Detection} \\',
         r' & \textbf{Event} & \textbf{Subjects} & \textbf{(approx)} & \textbf{Ch} & '
         r'\textbf{Montage} & \textbf{Channel} \\',
         r'\hline']
    _by_dir = {d[1]: d for d in DATASETS}
    for ds_dir in _T1_ORDER:
        d = _by_dir[ds_dir]
        key = d[0]
        v3, vf = d[2], d[3]
        cfg = json.load(open(DATASETS_DIR / ds_dir / 'dataset_config.json'))
        v3name = [k for k in cfg['variants'] if k != 'full'][0]
        chans = cfg.get('channel_presets', {}).get(v3name, [])
        det = cfg.get('detection_channel', '?')
        event = _EVENT_TYPE.get(ds_dir, '?')
        rj = DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / v3 / 'erpxttn_peak' / 'seed-1' / 'results.json'
        ns = 0; trials_per_subj = 0; n_full_ch = '?'
        if rj.exists():
            r = json.load(open(rj))
            ns = len(r.get('folds', []))
            counts = []
            for f in r['folds']:
                pf = rj.parent / f"predictions_{f['test_subject']}.npz"
                if pf.exists():
                    counts.append(len(np.load(pf)['labels']))
            trials_per_subj = int(round(np.mean(counts))) if counts else 0
        pf_full = sorted((DATASETS_DIR / ds_dir / 'results' / 'tmin0ms_tmax800ms' / vf
                          / 'erpxttn_peak' / 'seed-1').glob('prototypes_*.npz'))
        if pf_full:
            n_full_ch = np.load(pf_full[0])['proto_seg'].shape[1]
        ch_str = ', '.join(chans)
        label = _T1_LABELS.get(key, key)
        L.append(f'{label} & {event} & {ns} & {trials_per_subj} & {n_full_ch} '
                 f'& {ch_str} & {det} \\\\')
    L += [r'\hline', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


# --- Table 2: main 3-channel AUROC comparison ---
def write_main_auroc_table(out_path):
    L = [r'\begin{table}[t]', r'\centering',
         r'\setlength{\tabcolsep}{5pt}',
         r'\caption{AUROC (mean $\pm$ SD, LOSO) at the three-channel montage. '
         r'\textbf{Bold} marks the highest mean per dataset. '
         r'$\Delta$ = best baseline minus ERP-XTTN (performance gap, in AUROC). '
         r'Datasets ordered by mean AUROC across all five methods, descending. '
         r'Full-montage results are in Supplementary Table~\ref{sup:tab-fullmontage-auroc}; '
         r'per-subject AUROC values in Supplementary '
         r'Tables~\ref{sup:tab-persubj-hri}--\ref{sup:tab-persubj-n400}.}',
         r'\label{tab:auroc}',
         r'\begin{tabular}{lcccccc}', r'\toprule',
         r'Dataset & ERP-XTTN & EEGNet & EEG-Deformer & EPMN & xDAWN+RG & $\Delta$ \\',
         r'\midrule']
    rows = _difficulty_order()
    all_rounded = {n: [] for n, _ in _SUP_METHODS}
    all_deltas = []
    for d in rows:
        lab = d[0].replace(' ErrP', '')
        v3 = d[2]
        vals = {}
        for name, mdl in _SUP_METHODS:
            pj = _persubj_fusion(d[1], v3, mdl) if mdl == 'erpxttn_peak' else _persubj_forward(d[1], v3, mdl)
            m, s, n = _msd(pj)
            vals[name] = (round(m, 3) if m is not None else None,
                          round(s, 3) if s is not None else None)
        best = max((v[0] for v in vals.values() if v[0] is not None), default=None)
        cells = []
        for name, _ in _SUP_METHODS:
            m, s = vals[name]
            if m is None:
                cells.append('--')
            else:
                all_rounded[name].append(m)
                cell = f'{m:.3f} $\\pm$ {s:.3f}'
                if best is not None and abs(m - best) < 1e-4:
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
        xt_m = vals['ERP-XTTN'][0]
        bl = [vals[n][0] for n, _ in _SUP_METHODS if n != 'ERP-XTTN' and vals[n][0] is not None]
        if bl and xt_m is not None:
            delta = round(max(bl) - xt_m, 3)
            all_deltas.append(delta)
            L.append(f'{lab} & ' + ' & '.join(cells) + f' & \\textit{{{delta:.3f}}} \\\\')
        else:
            L.append(f'{lab} & ' + ' & '.join(cells) + r' & -- \\')
    L.append(r'\midrule')
    mean_cells = []
    for name, _ in _SUP_METHODS:
        v = all_rounded[name]
        m = round(float(np.mean(v)), 3) if v else None
        if m is None:
            mean_cells.append('--')
        else:
            mean_cells.append(f'{m:.3f}')
    best_mean = max(round(float(np.mean(all_rounded[n])), 3) for n, _ in _SUP_METHODS if all_rounded[n])
    for i, (name, _) in enumerate(_SUP_METHODS):
        v = all_rounded[name]
        if v and abs(round(float(np.mean(v)), 3) - best_mean) < 1e-4:
            mean_cells[i] = r'\textbf{' + mean_cells[i] + '}'
    mean_delta = round(float(np.mean(all_deltas)), 3) if all_deltas else None
    delta_cell = f'\\textbf{{{mean_delta:.3f}}}' if mean_delta is not None else '--'
    L.append(r'\textit{Mean} & ' + ' & '.join(mean_cells) + f' & {delta_cell} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _tex(out_path, L)


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

    print('Main-text tables (LaTeX)...')
    write_dataset_table(OUT_DIR / 'table1_datasets.tex')
    write_main_auroc_table(OUT_DIR / 'table2_auroc_3ch.tex')

    print('Paired advantage heatmap (Table 2 companion)...')
    fig_paired_heatmap(OUT_DIR / 'fig_paired_heatmap.png')

    print('Paired advantage table (Table 2 companion, LaTeX)...')
    write_paired_table(OUT_DIR / 'table_paired.tex')

    print('Supplementary tables (LaTeX)...')
    write_fullmontage_auroc_table(OUT_DIR / 'tableS1_fullmontage_auroc.tex')
    write_fullmontage_ba_table(OUT_DIR / 'tableS2_fullmontage_ba.tex')
    write_grounding_routing_table(OUT_DIR / 'tableS3_grounding_routing_full.tex', 'full-montage', 'S3')
    write_grounding_amp_table(OUT_DIR / 'tableS4_grounding_amp_full.tex', 'full-montage', 'S4')
    write_epmn_native_table(OUT_DIR / 'tableS5_epmn_native.tex')
    write_balacc_table(OUT_DIR / 'tableS6_balanced_accuracy.tex')
    write_persubject_auroc_tables(OUT_DIR / 'tableS7-S15_persubject_auroc.tex')
    write_wilcoxon_table(OUT_DIR / 'tableS16_wilcoxon.tex')
    write_corr_table(OUT_DIR / 'tableS17_tpfp_tptn_corr.tex')
    write_corr_perseed_table(OUT_DIR / 'tableS18_perseed_corr.tex')
    write_grounding_routing_table(OUT_DIR / 'tableS19_grounding_routing_3ch.tex', '3-channel', 'S19')
    write_grounding_amp_table(OUT_DIR / 'tableS20_grounding_amp_3ch.tex', '3-channel', 'S20')
    write_ablation_table(OUT_DIR / 'tableS21_ablation.tex')
    write_polarity_contrast_table(OUT_DIR / 'tableS22_polarity_contrast.tex')

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
