"""Benchmark single-trial CPU inference latency for all three baselines.

Matched settings: CPU, batch=1 (or single-trial for sklearn), 3-channel input,
0.8 s @ 256 Hz epochs. 20-iteration warm-up, 500-iteration measurement.

Reports include:
  - ERP-XTTN  — torch eval mode, torch.no_grad()
  - EEGNet    — torch eval mode, torch.no_grad()
  - xDAWN+RG  — fitted sklearn pipeline, predict_proba on single trial

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

from eegnet import EEGNet
from erpxttn import ERPXTTN
from xdawn_rg import XDawnRG


N_CHANNELS = 3
N_TIMES = 206        # 0.8 s @ 256 Hz, matches production cache
SFREQ = 256
N_PROTO = 4

WARMUP = 20
N_ITER = 500


def _cpu_brand() -> str:
    try:
        return subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return platform.processor() or platform.machine()


def _bench(model: torch.nn.Module, x: torch.Tensor,
           warmup: int = WARMUP, n_iter: int = N_ITER) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms
    a = np.array(times)
    return {
        'median': float(np.median(a)),
        'mean':   float(np.mean(a)),
        'p10':    float(np.percentile(a, 10)),
        'p90':    float(np.percentile(a, 90)),
        'min':    float(np.min(a)),
    }


def main() -> None:
    print('=' * 68)
    print('Batch-1 CPU forward-pass latency benchmark')
    print('=' * 68)
    print(f'CPU:        {_cpu_brand()}')
    print(f'Platform:   {platform.platform()}')
    print(f'Python:     {sys.version.split()[0]}')
    print(f'PyTorch:    {torch.__version__}')
    print(f'Threads:    {torch.get_num_threads()} (PyTorch default)')
    print(f'Input:      batch=1, channels={N_CHANNELS}, n_times={N_TIMES} '
          f'({N_TIMES/SFREQ*1000:.0f} ms @ {SFREQ} Hz)')
    print(f'Warm-up:    {WARMUP} iterations')
    print(f'Measured:   {N_ITER} iterations, eval mode, torch.no_grad()')
    print()

    x = torch.randn(1, N_CHANNELS, N_TIMES)

    # ---- ERP-XTTN (3ch, K=4) ----
    erp = ERPXTTN(n_channels=N_CHANNELS, n_times=N_TIMES,
                  max_k=N_PROTO, sfreq=SFREQ)
    # Populate prototype buffers (otherwise the model's set_prototypes path
    # gets exercised differently). These don't change the FLOP count of the
    # forward pass, just give it a non-trivial K x C x T tensor.
    erp.proto_raw.copy_(torch.randn(N_PROTO, N_CHANNELS, N_TIMES))
    patch_centers = torch.linspace(
        2, erp.N - 2, N_PROTO).round().long()
    erp.proto_center_patch_idx.copy_(patch_centers)
    erp_params = sum(p.numel() for p in erp.parameters() if p.requires_grad)
    erp_stats = _bench(erp, x)

    # ---- EEGNet (3ch, defaults) ----
    eeg = EEGNet(n_channels=N_CHANNELS, n_times=N_TIMES, srate=SFREQ)
    eeg_params = sum(p.numel() for p in eeg.parameters() if p.requires_grad)
    eeg_stats = _bench(eeg, x)

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
    # Warm-up
    for _ in range(WARMUP):
        _ = xdr.predict_proba(x_np)
    xdr_times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _ = xdr.predict_proba(x_np)
        t1 = time.perf_counter()
        xdr_times.append((t1 - t0) * 1000.0)
    xdr_arr = np.array(xdr_times)
    xdr_stats = {
        'median': float(np.median(xdr_arr)),
        'mean':   float(np.mean(xdr_arr)),
        'p10':    float(np.percentile(xdr_arr, 10)),
        'p90':    float(np.percentile(xdr_arr, 90)),
        'min':    float(np.min(xdr_arr)),
    }
    # "Parameters" isn't a clean concept for xDAWN+RG; report what's learned:
    # xDAWN filters (nfilter * n_channels) + LR weights/intercept.
    n_classes = 2
    n_xdawn = 4 * N_CHANNELS * n_classes  # 4 filters per class
    cov_dim = n_xdawn  # cov matrix is nxdawn x nxdawn (here 24x24)
    tangent_dim = cov_dim * (cov_dim + 1) // 2
    lr_params = tangent_dim + 1  # weights + bias
    xdr_params = n_xdawn + lr_params

    # ---- Report ----
    print(f'{"Model":<12} {"Params":>10}  {"Median (ms)":>12}  '
          f'{"Mean":>8}  {"P10":>7}  {"P90":>7}  {"Min":>7}')
    print('-' * 68)
    for name, params, s in [
        ('ERP-XTTN', erp_params, erp_stats),
        ('EEGNet',   eeg_params, eeg_stats),
        ('xDAWN+RG', xdr_params, xdr_stats),
    ]:
        print(f'{name:<12} {params:>10,}  {s["median"]:>12.3f}  '
              f'{s["mean"]:>8.3f}  {s["p10"]:>7.3f}  '
              f'{s["p90"]:>7.3f}  {s["min"]:>7.3f}')
    print()
    print('Note: xDAWN+RG "params" = xDAWN filter weights + LR tangent-space '
          'coefficients\n      (approximate; classical pipelines do not have '
          'a single canonical count).')


if __name__ == '__main__':
    main()
