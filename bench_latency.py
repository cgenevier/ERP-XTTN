"""Benchmark single-trial CPU inference latency for all three methods.

Matched settings: CPU, batch=1 (or single-trial for sklearn), 3-channel input,
0.8 s @ 256 Hz epochs. 20-iteration warm-up, 500-iteration measurement.

Each timed call is the FULL path that turns one raw epoch into a class
probability — i.e. what a real online BCI pays per trial:
  - ERP-XTTN    — normalize -> routing forward() -> amplitude compute_mf_channel()
                  -> Stage-2 logistic fusion (predict_proba). This is the
                  end-to-end two-factor decision, not just the routing forward.
  - EEGNet      — torch eval mode, torch.no_grad(), single forward.
  - EEG-Deformer — torch eval mode, torch.no_grad(), single forward.
  - EPMN        — forward -> squared distance to the 2 class prototypes ->
                  softmax (prototypes built once from support subjects).
  - xDAWN+RG    — fitted sklearn pipeline, predict_proba on single trial.

Usage:
    python bench_latency.py
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from eeg_deformer import EEGDeformer
from eegnet import EEGNet
from epmn import EPMN, build_prototypes, epmn_class_logits, squared_distances
from erpxttn import ERPXTTN, _fold_features
from xdawn_rg import XDawnRG


N_CHANNELS = 3
N_TIMES = 206        # 0.8 s @ 256 Hz, matches production cache
SFREQ = 256
N_PROTO = 4

WARMUP = 20
N_ITER = 500

# PyTorch intra-op thread counts to sweep. At batch=1 with tiny (C x T) tensors,
# single-thread is typically fastest (dispatch overhead dominates arithmetic);
# 4 threads is the library default. Both are reported so the reader sees the
# range and can pick the deployment-relevant one.
THREAD_COUNTS = (1, 4)


def _cpu_brand() -> str:
    try:
        return subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return platform.processor() or platform.machine()


def _stats(a: np.ndarray) -> dict:
    return {
        'median': float(np.median(a)),
        'mean':   float(np.mean(a)),
        'p10':    float(np.percentile(a, 10)),
        'p90':    float(np.percentile(a, 90)),
        'min':    float(np.min(a)),
    }


def _bench_call(fn, warmup: int = WARMUP, n_iter: int = N_ITER) -> dict:
    """Time a zero-argument callable (the full per-trial inference)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return _stats(np.array(times))


def _bench(model: torch.nn.Module, x: torch.Tensor,
           warmup: int = WARMUP, n_iter: int = N_ITER) -> dict:
    model.eval()
    with torch.no_grad():
        return _bench_call(lambda: model(x), warmup, n_iter)


def main() -> None:
    print('=' * 68)
    print('Batch-1 CPU end-to-end inference latency benchmark')
    print('=' * 68)
    print(f'CPU:        {_cpu_brand()}')
    print(f'Platform:   {platform.platform()}')
    print(f'Python:     {sys.version.split()[0]}')
    print(f'PyTorch:    {torch.__version__}')
    print(f'Threads:    sweeping {", ".join(str(t) for t in THREAD_COUNTS)} '
          f'(see per-table headers)')
    print(f'Input:      batch=1, channels={N_CHANNELS}, n_times={N_TIMES} '
          f'({N_TIMES/SFREQ*1000:.0f} ms @ {SFREQ} Hz)')
    print(f'Warm-up:    {WARMUP} iterations')
    print(f'Measured:   {N_ITER} iterations, eval mode, torch.no_grad()')
    print()

    x = torch.randn(1, N_CHANNELS, N_TIMES)

    # ---- ERP-XTTN (3ch, K=4), full two-factor path ----
    erp = ERPXTTN(n_channels=N_CHANNELS, n_times=N_TIMES,
                  max_k=N_PROTO, sfreq=SFREQ)
    # Populate the grounded prototype/template buffers via the production path
    # (peak detection -> whitening -> matched-filter templates). Fit-time data
    # doesn't affect the forward FLOP count; it just gives the readout a
    # non-trivial K prototypes to match against, as at inference time.
    n_fit_per = 60
    X_fit = np.random.default_rng(0).standard_normal(
        (2 * n_fit_per, N_CHANNELS, N_TIMES)).astype(np.float32)
    _bump = np.zeros(N_TIMES); _bump[N_TIMES // 3:N_TIMES // 2] = 1.0
    X_fit[:n_fit_per] += 0.5 * _bump[None, None, :]
    y_fit = np.concatenate([np.ones(n_fit_per), np.zeros(n_fit_per)]).astype(np.int64)
    erp.eval()
    erp.set_prototypes(torch.from_numpy(X_fit), torch.from_numpy(y_fit))

    # Per-channel normalization stats, exactly as a fold checkpoint stores them.
    nm = X_fit.mean(axis=(0, 2), keepdims=True)          # (1, C, 1)
    ns = X_fit.std(axis=(0, 2), keepdims=True) + 1e-7    # (1, C, 1)

    # Fit the Stage-2 LOSO-style logistic on the training set's two-factor
    # features [routing_logit, MF_kc, contrast]. Fit time is irrelevant to the
    # measured inference latency below.
    feat_fit = _fold_features(erp, ((X_fit - nm) / ns).astype(np.float32), 'cpu')
    fusion = LogisticRegression(max_iter=1000, class_weight='balanced').fit(
        feat_fit, y_fit)

    x_raw_np = x.numpy()  # (1, C, T), the single eval trial

    def _erp_two_factor() -> np.ndarray:
        # Full per-trial online path: normalize -> routing forward +
        # amplitude matched filter (via _fold_features) -> Stage-2 fusion.
        xn = ((x_raw_np - nm) / ns).astype(np.float32)
        feats = _fold_features(erp, xn, 'cpu')
        return fusion.predict_proba(feats)[:, 1]

    erp_params = sum(p.numel() for p in erp.parameters() if p.requires_grad)
    # Stage-2 adds the fusion logistic's coefficients (one per feature + bias).
    erp_fusion_params = int(fusion.coef_.size + fusion.intercept_.size)

    # ---- EEGNet (3ch, defaults) ----
    eeg = EEGNet(n_channels=N_CHANNELS, n_times=N_TIMES, srate=SFREQ)
    eeg_params = sum(p.numel() for p in eeg.parameters() if p.requires_grad)

    # ---- EEG-Deformer (3ch, defaults) ----
    # Single-logit CNN-transformer; forward() emits the decision logit directly,
    # so the timed path is one forward (as for EEGNet).
    dfm = EEGDeformer(n_channels=N_CHANNELS, n_times=N_TIMES, srate=SFREQ)
    dfm_params = sum(p.numel() for p in dfm.parameters() if p.requires_grad)

    # ---- EPMN (prototypical matching net) ----
    # End-to-end path: embed the trial, take squared distance to the two class
    # prototypes, softmax. Prototypes are built once from support subjects (a
    # fit-time step, like xDAWN's fit), so they are not part of the per-trial cost.
    epmn = EPMN(n_channels=N_CHANNELS, n_times=N_TIMES).eval()
    epmn_params = sum(p.numel() for p in epmn.parameters() if p.requires_grad)
    _rng_e = np.random.default_rng(0)
    support = [
        {k: torch.from_numpy(_rng_e.standard_normal(
            (N_CHANNELS, N_TIMES)).astype(np.float32)) for k in (0, 1)}
        for _ in range(4)
    ]
    with torch.no_grad():
        epmn_protos = build_prototypes(epmn, support, 'cpu')  # (2, D), one-time

    def _epmn_predict() -> torch.Tensor:
        emb = epmn(x)                                   # (1, D)
        d = squared_distances(emb, epmn_protos)         # (1, 2)
        return torch.softmax(epmn_class_logits(d), dim=1)

    epmn.eval()

    # ---- xDAWN+RG (sklearn pipeline) ----
    # Fit on a synthetic class-balanced training set (60 trials of each class)
    # — fit-time tuning doesn't affect inference latency, which is what we
    # care about here.
    rng = np.random.default_rng(0)
    n_train_per = 60
    X_train = rng.standard_normal(
        (2 * n_train_per, N_CHANNELS, N_TIMES)).astype(np.float64)
    # Mild class-separating signal so xDAWN has something to fit
    bump = np.zeros(N_TIMES)
    bump[N_TIMES // 3:N_TIMES // 2] = 1.0
    X_train[:n_train_per] += 0.5 * bump[None, None, :]
    y_train = np.concatenate([np.ones(n_train_per), np.zeros(n_train_per)])
    xdr = XDawnRG(nfilter=4).fit(X_train, y_train)

    x_np = x.numpy().astype(np.float64)
    # "Parameters" isn't a clean concept for xDAWN+RG; report what's learned:
    # xDAWN filters (nfilter * n_channels) + LR weights/intercept.
    n_classes = 2
    n_xdawn = 4 * N_CHANNELS * n_classes  # 4 filters per class
    cov_dim = n_xdawn  # cov matrix is nxdawn x nxdawn (here 24x24)
    tangent_dim = cov_dim * (cov_dim + 1) // 2
    lr_params = tangent_dim + 1  # weights + bias
    xdr_params = n_xdawn + lr_params

    def measure_all() -> list:
        """Time every method's full per-trial path at the current thread count."""
        with torch.no_grad():
            erp_s = _bench_call(_erp_two_factor)
            eeg_s = _bench(eeg, x)
            dfm_s = _bench(dfm, x)
            epmn_s = _bench_call(_epmn_predict)
        xdr_s = _bench_call(lambda: xdr.predict_proba(x_np))
        return [
            ('ERP-XTTN',     erp_params,  erp_s),
            ('EEGNet',       eeg_params,  eeg_s),
            ('EEG-Deformer', dfm_params,  dfm_s),
            ('EPMN',         epmn_params, epmn_s),
            ('xDAWN+RG',     xdr_params,  xdr_s),
        ]

    # ---- Report: one table per thread count ----
    for nth in THREAD_COUNTS:
        torch.set_num_threads(nth)
        rows = measure_all()
        print(f'PyTorch intra-op threads = {nth}')
        print(f'{"Model":<14} {"Params":>10}  {"Median (ms)":>12}  '
              f'{"Mean":>8}  {"P10":>7}  {"P90":>7}  {"Min":>7}')
        print('-' * 74)
        for name, params, s in rows:
            print(f'{name:<14} {params:>10,}  {s["median"]:>12.3f}  '
                  f'{s["mean"]:>8.3f}  {s["p10"]:>7.3f}  '
                  f'{s["p90"]:>7.3f}  {s["min"]:>7.3f}')
        print()
    print(f'Note: ERP-XTTN latency is the full two-factor path (normalize -> '
          f'routing forward ->\n      amplitude matched filter -> Stage-2 '
          f'logistic fusion). The "Params" column is the\n      network\'s '
          f'trainable weights; Stage-2 adds {erp_fusion_params} fusion '
          f'coefficients on top.')
    print('Note: xDAWN+RG "params" = xDAWN filter weights + LR tangent-space '
          'coefficients\n      (approximate; classical pipelines do not have '
          'a single canonical count).')


if __name__ == '__main__':
    main()
