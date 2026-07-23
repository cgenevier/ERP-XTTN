"""ERP-XTTN — two-factor grounded cross-attention ERP classifier.

Peak-unit / grounded-readout architecture.

Two grounded decision pathways:

  (1) routing / morphology — learned QK cross-attention over the trial's
      detected *peak units* to K difference-wave *prototype* templates, read out
      by a FIXED grounded bilinear

          logit = match_scale * Σ_{p,k} a[p,k] · m[p,k] / (#valid peaks)

      where a[p,k] is the learned attention of peak p to prototype k and m[p,k]
      is the cosine similarity of the whitened peak segment to the whitened
      prototype window. The match m is a fixed function of the physiology (no
      learned parameters); only the routing a and the scalar match_scale learn.
      That is what makes the prototypes load-bearing by construction.

  (2) amplitude / matched-filter — per-component *raw* projections of the trial
      onto each template, exposed by compute_mf() plus channel-resolved and
      bipolar contrast decompositions from compute_mf_channel(). These grounded
      amplitude factors are fused with the routing logit OUTSIDE the model, via
      leave-one-subject-out logistic regression (two-stage late fusion).

Two widths are in play and MUST NOT be conflated:
  * patch_width (=8) — the width every unit is resampled to for the *embedding*
    that feeds the learned routing attention a.
  * the prototype's native window width (e−s) — the width the *match* m and its
    whitening run at (joint spatiotemporal Σ^(−1/2) over C·(e−s)). A ~130 ms
    component at 256 Hz is matched at ~33 samples, not 8.

Terminology:
  * units / tokens  = the trial's detected peaks (variable count, ≤ max_peaks).
  * prototypes (K)  = component templates from the training-fold grand-average
    difference wave. "K" always refers to the prototypes.

Public interface: input (B, C, T); forward() returns (logit (B, 1), aux) where
aux holds the grounded intermediates {a, m, mask, center, bounds} for dumps and
interpretability figures.
"""

import json
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Canonical detection channel fallback (configs set this per dataset).
DETECT_CHANNEL = "Cz"

# Minimum latency (ms) for the first peak — avoids locking onto early
# noise/artifact before the physiological ERP response.
MIN_P1_LATENCY_MS = 50.0

# Gaussian smoothing sigma for TRIAL peak detection + inflection windows.
# Canonical value is 3.0 (matchedcos match_smooth); prototypes use 2.0. A
# smaller value over-detects noise-wiggle peaks and dilutes the Σ a·m routing.
TRIAL_SMOOTH_SIGMA = 3.0


def ms_to_sample(ms: float, sfreq: float = 256.0, tmin: float = 0.0) -> int:
    """Convert milliseconds to sample index."""
    return int(round((ms / 1000.0 - tmin) * sfreq))


def channel_pairs(n_channels: int) -> list[tuple[int, int]]:
    """Stable all-pairs channel contrasts (i, j) with i < j."""
    return [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]


# ──────────────────────────────────────────────────────────────────────
# Signal helpers (shared by prototype detection and per-trial tokenization)
# ──────────────────────────────────────────────────────────────────────

def _smooth_signal(diff_signal: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth a 1-D signal for robust peak/zero-crossing detection.

    Used for PROTOTYPE detection (scipy default, 4σ truncation)."""
    return gaussian_filter1d(diff_signal, sigma=sigma)


def _smooth_trial(sig: np.ndarray, sigma: float) -> np.ndarray:
    """Trial-peak smoothing — bit-identical to matchedcos `_smooth_time`: a manual
    Gaussian kernel truncated at 3σ (radius round(3σ)), reflect-padded, run as a
    float32 torch conv1d (same op/dtype as matchedcos, so near-zero second-
    derivatives flip identically and inflection windows match). scipy's
    gaussian_filter1d truncates at 4σ and rounds differently — the old drift."""
    return _smooth_trial_batch(sig[None, :], sigma)[0]


def _smooth_trial_batch(sig2d: np.ndarray, sigma: float) -> np.ndarray:
    """Batched `_smooth_trial`: (B, T) → (B, T) in one torch conv. matchedcos
    smooths the whole batch's detection channel at once, so batching here is both
    faster AND bit-identical (per-trial torch convs in the loop were ~10× slower)."""
    rad = max(1, int(round(3 * sigma)))
    t = torch.arange(-rad, rad + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    x = torch.from_numpy(np.ascontiguousarray(sig2d, dtype=np.float32)).unsqueeze(1)
    xr = F.pad(x, (rad, rad), mode="reflect")
    return F.conv1d(xr, k).squeeze(1).numpy()


def _find_zero_crossings(signal: np.ndarray) -> np.ndarray:
    """Sample indices (last sample before each sign change)."""
    signs = np.sign(signal)
    for i in range(len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1] if i > 0 else 1
    return np.where(np.diff(signs) != 0)[0]


def _resample_to_width(seg: np.ndarray, width: int) -> np.ndarray:
    """Linearly resample a (C, w) segment to (C, width) along time."""
    C, w = seg.shape
    if w == width:
        return seg.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, num=w)
    x_new = np.linspace(0.0, 1.0, num=width)
    out = np.empty((C, width), dtype=np.float32)
    for c in range(C):
        out[c] = np.interp(x_new, x_old, seg[c])
    return out


def _resample_batch(X: np.ndarray, bs: np.ndarray, los: np.ndarray,
                    his: np.ndarray, width: int) -> np.ndarray:
    """Vectorized linear resample of N variable-width windows to a common width.

    For each i, resamples X[bs[i], :, los[i]:his[i]] (a (C, w_i) segment) to
    (C, width) with the SAME linear interpolation as _resample_to_width, but
    gathers all N units and all channels in one op. Returns (N, C, width).
    Bit-for-bit equivalent to looping _resample_to_width over units (float64
    blend, matching np.interp); this replaces ~P*K per-unit np.interp calls.
    """
    N = len(bs)
    C, T = X.shape[1], X.shape[2]
    if N == 0:
        return np.zeros((0, C, width), dtype=np.float32)
    w = (his - los).astype(np.float64)                          # (N,) native widths
    if width > 1:
        tgrid = np.arange(width, dtype=np.float64) / (width - 1.0)
    else:
        tgrid = np.zeros(1, dtype=np.float64)
    frac_idx = tgrid[None, :] * (w[:, None] - 1.0)              # (N, width) in [0, w-1]
    abs_pos = los[:, None].astype(np.float64) + frac_idx        # (N, width) absolute
    flf = np.floor(abs_pos)
    fr = abs_pos - flf                                          # (N, width) in [0,1) float64
    fl = np.clip(flf.astype(np.int64), 0, T - 1)               # (N, width)
    ce = np.clip(flf.astype(np.int64) + 1, 0, T - 1)
    bexp = bs[:, None, None]                                    # (N,1,1)
    cexp = np.arange(C)[None, :, None]                          # (1,C,1)
    lo_v = X[bexp, cexp, fl[:, None, :]].astype(np.float64)     # (N, C, width)
    hi_v = X[bexp, cexp, ce[:, None, :]].astype(np.float64)     # (N, C, width)
    out = lo_v * (1.0 - fr[:, None, :]) + hi_v * fr[:, None, :]
    return out.astype(np.float32)


def detect_trial_peaks(
    signal: np.ndarray, sfreq: float,
    prominence: float,
    max_peaks: int,
    min_distance_ms: float = 40.0,
    smooth_sigma: float = TRIAL_SMOOTH_SIGMA,
    presmoothed: np.ndarray = None,
) -> list[tuple[int, str]]:
    """Detect a trial's peak *units* on its detection channel.

    The tokens the router attends *from*: the top-`max_peaks` most prominent
    deflections — positive and negative pooled, **mixed polarity, ranked by
    prominence, with NO alternating-polarity constraint** (that constraint is for
    the prototype chain, not trial units) and **no latency floor** (which would
    drop legitimate early units). Kept peaks are restored to temporal order.
    Returns (sample_index, polarity) by time.
    """
    min_distance = max(1, int(round(min_distance_ms / 1000.0 * sfreq)))
    smoothed = _smooth_trial(signal, smooth_sigma) if presmoothed is None else presmoothed

    pos_peaks, pos_props = find_peaks(
        smoothed, prominence=prominence, distance=min_distance)
    neg_peaks, neg_props = find_peaks(
        -smoothed, prominence=prominence, distance=min_distance)

    cand = [(int(i), 'pos', float(p))
            for i, p in zip(pos_peaks, pos_props['prominences'])]
    cand += [(int(i), 'neg', float(p))
             for i, p in zip(neg_peaks, neg_props['prominences'])]
    if not cand:
        return []

    # Top-max_peaks by prominence (mixed polarity), then restore temporal order.
    cand.sort(key=lambda t: t[2], reverse=True)
    keep = sorted(cand[:max_peaks], key=lambda t: t[0])
    return [(idx, pol) for idx, pol, _ in keep]


def _find_inflections(smoothed_signal: np.ndarray) -> np.ndarray:
    """Inflection sample indices (second-derivative sign changes).

    The right boundary for *single-trial* peaks: unlike a difference wave, a
    single-trial signal does not ride around zero, so its deflections are bounded
    by their flanking inflections (the shoulders of the bump), not zero-crossings.
    """
    return np.where(np.diff(np.sign(np.diff(smoothed_signal, n=2))) != 0)[0] + 1


def _windows_from_boundaries(peak_indices, boundaries, T, min_w, max_w):
    """Expand each peak to its flanking boundary indices, clamp width, no overlap."""
    windows = []
    for i, peak in enumerate(peak_indices):
        before = boundaries[boundaries < peak]
        left = int(before[-1]) + 1 if len(before) > 0 else 0
        after = boundaries[boundaries >= peak]
        right = int(after[0]) + 1 if len(after) > 0 else T

        if i > 0:
            left = max(left, windows[-1][1])

        if right - left < min_w:
            deficit = min_w - (right - left)
            el = deficit // 2
            left = max(left - el, 0 if i == 0 else windows[-1][1])
            right = min(right + (deficit - el), T)
        if right - left > max_w:
            excess = (right - left - max_w) // 2
            left += excess
            right = left + max_w

        windows.append((max(left, 0), min(right, T)))
    return windows


def build_windows_from_zero_crossings(
    peak_indices: list[int], smoothed_signal: np.ndarray,
    min_window_ms: float = 40.0,
    max_window_ms: float = 200.0,
    sfreq: float = 256.0,
) -> list[tuple[int, int]]:
    """Windows bounded by flanking zero-crossings — correct for the difference
    wave (which crosses zero between components). Used for the PROTOTYPES."""
    T = len(smoothed_signal)
    min_w = int(round(min_window_ms / 1000.0 * sfreq))
    max_w = int(round(max_window_ms / 1000.0 * sfreq))
    return _windows_from_boundaries(
        peak_indices, _find_zero_crossings(smoothed_signal), T, min_w, max_w)


def build_windows_from_inflections(
    peak_indices: list[int], smoothed_signal: np.ndarray,
    sfreq: float = 256.0,
) -> list[tuple[int, int]]:
    """Windows bounded by the BARE flanking inflection points — for single-TRIAL
    peaks. Each window is just the inflection before and the inflection at/after
    the peak (a ≥2-sample floor guards degenerate cases). Unlike the prototype
    builder there is NO min/max width clamp and NO no-overlap constraint: a
    single trial does not ride around zero, so a unit's natural extent is exactly
    its flanking second-derivative shoulders.
    """
    T = len(smoothed_signal)
    infl = _find_inflections(smoothed_signal)
    windows = []
    for peak in peak_indices:
        # matchedcos exactly: lower bound from side='left', UPPER bound from
        # side='right' — so a peak that coincides with an inflection takes the
        # NEXT inflection as its right shoulder (a single 'left' search would
        # take the coincident one, one sample low).
        posL = int(np.searchsorted(infl, peak, side='left'))
        posR = int(np.searchsorted(infl, peak, side='right'))
        lo = int(infl[posL - 1]) if posL > 0 else 0
        hi = int(infl[posR]) if posR < len(infl) else T
        if hi - lo < 2:
            hi = min(T, lo + 2)
        windows.append((max(lo, 0), min(hi, T)))
    return windows


def _compute_whitener(win_flat: np.ndarray, y: np.ndarray,
                      shrink: float = 0.9, eig_floor: float = 1e-8) -> np.ndarray:
    """Joint spatiotemporal Σ^(−1/2) from pooled within-class residual covariance.

    Args:
        win_flat: (N_epochs, D) window data flattened to D = C · (e−s).
        y:        (N_epochs,) class labels.
        shrink:   Σ = shrink·cov + (1−shrink)·(tr(cov)/D)·I  (fixed, not LW).

    Returns:
        (D, D) symmetric Σ^(−1/2).
    """
    N, D = win_flat.shape
    classes = np.unique(y)
    resid = win_flat.copy()
    for c in classes:
        m = y == c
        resid[m] -= win_flat[m].mean(axis=0, keepdims=True)
    dof = max(1, N - len(classes))
    cov = resid.T @ resid / dof
    cov = shrink * cov + (1.0 - shrink) * (np.trace(cov) / D) * np.eye(D)
    w, V = eigh(cov)
    w = np.clip(w, eig_floor, None)
    return (V * (w ** -0.5)) @ V.T


# ──────────────────────────────────────────────────────────────────────
# On-device (GPU) tokenizer — batched Torch equivalents of the NumPy path.
# Every op stays on x.device; no host round-trip. Validated bit-parity vs the
# NumPy/scipy pipeline: peaks/windows/centers identical, match m to ~1e-7.
# ──────────────────────────────────────────────────────────────────────

def _t_smooth(s: torch.Tensor, sigma: float) -> torch.Tensor:
    """(B,T) Gaussian smooth, reflect-pad — same kernel/op as _smooth_trial_batch."""
    rad = max(1, int(round(3 * sigma)))
    tt = torch.arange(-rad, rad + 1, dtype=torch.float32, device=s.device)
    k = torch.exp(-0.5 * (tt / sigma) ** 2); k = (k / k.sum()).view(1, 1, -1)
    xr = F.pad(s.unsqueeze(1), (rad, rad), mode="reflect")
    return F.conv1d(xr, k).squeeze(1)


def _t_local_maxima(s: torch.Tensor) -> torch.Tensor:
    """(B,T) bool strict interior local maxima."""
    B, T = s.shape
    m = torch.zeros(B, T, dtype=torch.bool, device=s.device)
    m[:, 1:-1] = (s[:, 1:-1] > s[:, :-2]) & (s[:, 1:-1] > s[:, 2:])
    return m


def _t_prominences(s: torch.Tensor) -> torch.Tensor:
    """(B,T) scipy peak_prominences for every position: height minus the higher of
    the two flanking bases (min from the peak to the first strictly-higher sample)."""
    B, T = s.shape; dev = s.device
    idx = torch.arange(T, device=dev)
    ig = idx.view(1, 1, T).expand(B, T, T)
    higher = s.unsqueeze(1) > s.unsqueeze(2)                # [b,p,i]: s[i] > s[p]
    il = idx.view(1, 1, T) < idx.view(1, T, 1)
    ir = idx.view(1, 1, T) > idx.view(1, T, 1)
    ngl = torch.where(higher & il, ig, torch.full_like(ig, -1)).amax(2)
    ngr = torch.where(higher & ir, ig, torch.full_like(ig, T)).amin(2)
    si = s.unsqueeze(1).expand(B, T, T)
    INF = torch.tensor(float('inf'), device=dev)
    lb = (ig > ngl.unsqueeze(2)) & (idx.view(1, 1, T) <= idx.view(1, T, 1))
    rb = (ig < ngr.unsqueeze(2)) & (idx.view(1, 1, T) >= idx.view(1, T, 1))
    left_base = torch.where(lb, si, INF).amin(2)
    right_base = torch.where(rb, si, INF).amin(2)
    return s - torch.maximum(left_base, right_base)


def _t_nms(peakmask: torch.Tensor, height: torch.Tensor, distance: int) -> torch.Tensor:
    """1D NMS by height (scipy _select_by_peak_distance): keep a peak iff no strictly
    taller *kept* peak within `distance`; iterated to a fixpoint (greedy cascade)."""
    B, T = peakmask.shape; dev = peakmask.device
    idx = torch.arange(T, device=dev)
    within = (idx.view(1, T, 1) - idx.view(1, 1, T)).abs() < distance
    h = height.unsqueeze(1); hp = height.unsqueeze(2)
    taller = (h > hp) | ((h == hp) & (idx.view(1, 1, T) < idx.view(1, T, 1)))
    keep = peakmask.clone()
    for _ in range(T):
        killers = within & taller & keep.unsqueeze(1) & peakmask.unsqueeze(2)
        newkeep = peakmask & ~(killers.any(2) & peakmask)
        if torch.equal(newkeep, keep):
            break
        keep = newkeep
    return keep


def _t_find_peaks(s: torch.Tensor, prominence: float, distance: int):
    """(peakmask (B,T) bool, prom (B,T)) after distance-then-prominence filter."""
    kept = _t_nms(_t_local_maxima(s), s, distance)
    prom = _t_prominences(s)
    return kept & (prom >= prominence), prom


def _t_nearest(mask_bt: torch.Tensor, want_left: bool) -> torch.Tensor:
    """(B,T): for each position, nearest index on one side with mask set (else 0/T)."""
    B, T = mask_bt.shape; dev = mask_bt.device
    idx = torch.arange(T, device=dev)
    ig = idx.view(1, 1, T).expand(B, T, T)
    if want_left:
        sel = (idx.view(1, 1, T) < idx.view(1, T, 1)) & mask_bt.unsqueeze(1)
        return torch.where(sel, ig, torch.full_like(ig, -1)).amax(2).clamp_min(0)
    sel = (idx.view(1, 1, T) > idx.view(1, T, 1)) & mask_bt.unsqueeze(1)
    return torch.where(sel, ig, torch.full_like(ig, T)).amin(2)


def _t_resample(X: torch.Tensor, bs, lo, hi, W: int) -> torch.Tensor:
    """(N,C,W) linear resample of X[bs[i],:,lo[i]:hi[i]] to width W (matches
    _resample_batch's index math; float32 vs the NumPy float64 blend → ~1e-6)."""
    N = bs.shape[0]; C, T = X.shape[1], X.shape[2]
    if N == 0:
        return torch.zeros(N, C, W, device=X.device)
    w = (hi - lo).float()
    tgrid = (torch.arange(W, device=X.device).float() / (W - 1)) if W > 1 \
        else torch.zeros(1, device=X.device)
    abs_pos = lo[:, None].float() + tgrid[None, :] * (w[:, None] - 1.0)
    fl = torch.floor(abs_pos); fr = abs_pos - fl
    fl = fl.long().clamp(0, T - 1); ce = (fl + 1).clamp(0, T - 1)
    Xg = X[bs]
    lo_v = Xg.gather(2, fl[:, None, :].expand(N, C, W))
    hi_v = Xg.gather(2, ce[:, None, :].expand(N, C, W))
    return lo_v * (1 - fr[:, None, :]) + hi_v * fr[:, None, :]


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Project a (·, C, patch_width) peak/prototype segment to d_model."""

    def __init__(self, n_channels: int, patch_width: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_channels * patch_width, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(-2))


class ERPXTTN(nn.Module):
    """Two-factor grounded ERP-XTTN (peak units + native-width matched readout).

    Input:  (B, C, T)
    Output: (logit (B, 1), aux dict)

    Call set_prototypes() before the first forward pass in each fold.
    """

    def __init__(self, n_channels: int, n_times: int,
                 channel_names: list[str] = None,
                 d_model: int = 64, num_heads: int = 4,
                 patch_width: int = 8, dropout: float = 0.3,
                 sfreq: float = 256.0, tmin: float = 0.0,
                 peak_prominence: float = 0.02,
                 min_window_ms: float = 40.0,
                 max_window_ms: float = 200.0,
                 detection_channel: str = None,
                 max_k: int = 4,
                 max_peaks: int = 14,
                 use_self_attn: bool = True,
                 whiten_shrink: float = 0.9):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.n_channels = n_channels
        self.n_times = n_times
        self.channel_names = (list(channel_names) if channel_names is not None
                              else [f"ch{i}" for i in range(n_channels)])
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.patch_width = patch_width
        self.use_self_attn = use_self_attn
        self.max_k = max_k
        self.K = max_k                 # initial proto count; re-derived per fold
        self.peak_prominence = peak_prominence
        self.max_peaks = max_peaks
        self.sfreq = sfreq
        self.tmin = tmin
        self.min_window_ms = min_window_ms
        self.max_window_ms = max_window_ms
        self.whiten_shrink = whiten_shrink
        # Upper bound on a prototype window width (samples) → whitener size.
        self.w_max = max(1, int(round(max_window_ms / 1000.0 * sfreq)))
        self.n_grid = max(1, n_times // patch_width)

        # Resolve detection channel by name.
        detect_name = detection_channel or DETECT_CHANNEL
        self.detect_name = detect_name
        if channel_names is not None and detect_name in channel_names:
            self.detect_ch = channel_names.index(detect_name)
        else:
            self.detect_ch = min(1, n_channels - 1)
            if channel_names is not None:
                logging.warning(
                    f"Detection channel '{detect_name}' not found in "
                    f"{channel_names}; falling back to index {self.detect_ch}")

        self.detected_windows_ms: list[tuple[float, float]] = []

        # Shared patch embedding for peak units and prototype key segments.
        self.patch_embed = PatchEmbedding(n_channels, patch_width, d_model)

        # Frozen random positional grid, interpolated at fractional unit centres
        # (parameter-free: registered as a buffer, never trained).
        # LEARNABLE positional grid (canonical: matchedcos base has freeze_frontend=False).
        # Freezing this was a clean-room regression — it starves the routing attention.
        self.pos_embed = nn.Parameter(torch.randn(self.n_grid, d_model) * 0.02)

        # Cross-attention (QK only — peaks query, prototypes key).
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)

        # Learned scalar calibrating the grounded logit's range (cannot change
        # WHICH prototype a peak matches — only the overall scale).
        # Init 10.0 (matchedcos): Σ_k a·m per peak is bounded ≈[-1,1], so scale=1
        # kept logits in ≈[-1,1] → sigmoid stuck ≈[0.27,0.73] → weak early gradient
        # through the routing attention (∂L/∂a ∝ scale·m). match_bias centers the
        # magnitude-pooled readout (AUROC-neutral but affects the BCE fit).
        self.match_scale = nn.Parameter(torch.tensor(10.0))
        self.match_bias = nn.Parameter(torch.zeros(1))
        self.dropout_layer = nn.Dropout(dropout)

        # Self-attention over peak tokens (ablatable via use_self_attn).
        self.sa_ln = nn.LayerNorm(d_model)
        self.sa_W_q = nn.Linear(d_model, d_model)
        self.sa_W_k = nn.Linear(d_model, d_model)
        self.sa_W_v = nn.Linear(d_model, d_model)
        self.sa_out_proj = nn.Linear(d_model, d_model)
        self.sa_dropout = nn.Dropout(dropout)

        self._register_proto_buffers(self.K)

    def _register_proto_buffers(self, k: int):
        """(Re)register the per-fold grounded buffers for K prototypes."""
        C, pw, Dm = self.n_channels, self.patch_width, self.n_channels * self.w_max
        # Key-embedding segment (width patch_width) for the routing pathway.
        self.register_buffer("proto_seg", torch.zeros(k, C, pw))
        self.register_buffer("proto_center", torch.zeros(k))
        # Native-width match buffers (padded to w_max; proto_w gives the true
        # width). whitener/proto_white use the leading C·(e−s) block.
        self.register_buffer("proto_w", torch.zeros(k, dtype=torch.long))
        self.register_buffer("whitener", torch.zeros(k, Dm, Dm))
        self.register_buffer("proto_white", torch.zeros(k, Dm))
        self.register_buffer("proto_white_norm", torch.ones(k))
        # Raw full-length template + window + norm for the amplitude MF factor.
        self.register_buffer("mf_template", torch.zeros(k, C, self.n_times))
        self.register_buffer("mf_window", torch.zeros(k, 2, dtype=torch.long))
        self.register_buffer("mf_template_norm", torch.ones(k))

    def _resize_K(self, k: int):
        self.K = k
        self._register_proto_buffers(k)
        self.to(self.match_scale.device)

    # ── prototype construction ─────────────────────────────────────────
    def _auto_detect_peaks(self, diff_signal: np.ndarray, sfreq: float):
        """Prototype peaks, matchedcos peak_mode='auto': the top-max_k most
        prominent pos/neg deflections — sign-consistent (pos where smoothed>0,
        neg where <0), past MIN_P1 latency, 80 ms min-distance, ranked by
        prominence. NO polarity-alternation constraint. Returns [(idx, pol)]."""
        min_distance = max(1, int(round(80.0 / 1000.0 * sfreq)))
        min_latency = int(round(MIN_P1_LATENCY_MS / 1000.0 * sfreq))
        smoothed = _smooth_signal(diff_signal, sigma=2.0)
        pos, pp = find_peaks(smoothed, prominence=self.peak_prominence, distance=min_distance)
        neg, npp = find_peaks(-smoothed, prominence=self.peak_prominence, distance=min_distance)
        cand = [(int(i), 'pos', float(p)) for i, p in zip(pos, pp['prominences'])
                if i >= min_latency and smoothed[i] > 0]
        cand += [(int(i), 'neg', float(p)) for i, p in zip(neg, npp['prominences'])
                 if i >= min_latency and smoothed[i] < 0]
        cand.sort(key=lambda t: t[2], reverse=True)
        keep = sorted(cand[:self.max_k], key=lambda t: t[0])
        return [(i, pol) for i, pol, _ in keep]

    def set_prototypes(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Detect prototype windows and build grounded templates + whiteners.

        Per prototype k: the width-patch_width shape segment for the routing key
        embedding (proto_seg); the joint spatiotemporal whitener Σ_k^(−1/2) at
        the window's NATIVE width from the training fold; the whitened native
        template (for the cosine match m); and the raw full-window template with
        its norm (for the amplitude matched filter MF).
        """
        Xtr = X_train.detach().cpu().numpy()
        ytr = y_train.detach().cpu().numpy()

        err = Xtr[ytr == 1].mean(axis=0)
        cor = Xtr[ytr == 0].mean(axis=0)
        diff = err - cor
        sig = diff[self.detect_ch]
        smoothed = _smooth_signal(sig, sigma=2.0)

        # matchedcos peak_mode='auto': top-max_k most-prominent pos/neg
        # deflections (NO polarity-alternation constraint — that was a clean-room
        # deviation that dropped/mis-placed prototypes on non-alternating waves).
        peaks = self._auto_detect_peaks(sig, self.sfreq)
        if not peaks:
            raise RuntimeError(
                f"Auto peak detection found no peaks on the grand-average "
                f"difference wave (prominence={self.peak_prominence}). Check "
                f"preprocessing/detection channel.")

        peak_idx = [p[0] for p in peaks]
        raw_windows = build_windows_from_zero_crossings(
            peak_idx, smoothed, self.min_window_ms, self.max_window_ms, self.sfreq)
        # Auto mode: drop peaks whose zero-crossing window doesn't contain them
        # (K shrinks). This is also what avoids the degenerate zero-width window
        # that used to crash set_prototypes on N400.
        keep = [(p, w) for p, w in zip(peaks, raw_windows) if w[0] <= p[0] <= w[1]]
        windows = [w for _, w in keep]

        if len(windows) != self.K:
            self._resize_K(len(windows))

        self.detected_windows_ms = [
            (round(s / self.sfreq * 1000 + self.tmin * 1000, 1),
             round(e / self.sfreq * 1000 + self.tmin * 1000, 1))
            for s, e in windows]

        C, pw, Dm = self.n_channels, self.patch_width, self.n_channels * self.w_max
        proto_seg = np.zeros((self.K, C, pw), dtype=np.float32)
        proto_w = np.zeros(self.K, dtype=np.int64)
        whitener = np.zeros((self.K, Dm, Dm), dtype=np.float32)
        proto_white = np.zeros((self.K, Dm), dtype=np.float32)
        proto_white_norm = np.ones(self.K, dtype=np.float32)
        mf_template = np.zeros((self.K, C, self.n_times), dtype=np.float32)
        mf_win = np.zeros((self.K, 2), dtype=np.int64)
        mf_norm = np.ones(self.K, dtype=np.float32)
        centers = np.zeros(self.K, dtype=np.float32)

        for k, (s, e) in enumerate(windows):
            w = min(e - s, self.w_max)
            e = s + w
            d = C * w

            # (a) key-embedding segment, resampled to patch_width.
            proto_seg[k] = _resample_to_width(diff[:, s:e], pw)

            # (b) native-width joint whitener from the training epochs' window.
            win_epochs = np.stack(
                [Xtr[n, :, s:e].reshape(-1) for n in range(Xtr.shape[0])], axis=0)
            W = _compute_whitener(win_epochs, ytr, shrink=self.whiten_shrink)
            whitener[k, :d, :d] = W
            proto_w[k] = w
            pw_vec = W @ diff[:, s:e].reshape(-1)          # whitened native template
            proto_white[k, :d] = pw_vec
            proto_white_norm[k] = float(np.linalg.norm(pw_vec) + 1e-8)

            # (c) raw full-window template + norm for the amplitude MF factor.
            mf_template[k, :, s:e] = diff[:, s:e]
            mf_win[k] = (s, e)
            mf_norm[k] = float(np.linalg.norm(diff[:, s:e]) + 1e-8)
            centers[k] = 0.5 * (s + e)

        dev = self.match_scale.device
        self.proto_seg = torch.from_numpy(proto_seg).to(dev)
        self.proto_center = torch.from_numpy(centers).to(dev)
        self.proto_w = torch.from_numpy(proto_w).to(dev)
        self.whitener = torch.from_numpy(whitener).to(dev)
        self.proto_white = torch.from_numpy(proto_white).to(dev)
        self.proto_white_norm = torch.from_numpy(proto_white_norm).to(dev)
        self.mf_template = torch.from_numpy(mf_template).to(dev)
        self.mf_window = torch.from_numpy(mf_win).to(dev)
        self.mf_template_norm = torch.from_numpy(mf_norm).to(dev)

    # ── positional encoding ─────────────────────────────────────────────
    def _interp_pos(self, centers: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of the frozen positional grid at unit centres.

        centers: (...,) sample indices (float). Returns (..., d_model).
        """
        g = (centers.float() / self.patch_width).clamp(0, self.n_grid - 1)
        lo = g.floor().long()
        hi = torch.clamp(lo + 1, max=self.n_grid - 1)
        frac = (g - lo.float()).unsqueeze(-1)
        return self.pos_embed[lo] * (1 - frac) + self.pos_embed[hi] * frac

    # ── tokenization + grounded match ───────────────────────────────────
    def _tokenize_and_match(self, x: torch.Tensor):
        """Per-trial peak units + the fixed grounded match m.

        Runs on the (possibly augmented) x. For each detected peak: the width-8
        embedding segment (routing pathway) and the native-width whitened-cosine
        match m[p,k] to every prototype (match pathway). m carries no gradient.

        Fully on-device (no host round-trip): batched Torch peak detection
        (local maxima → distance NMS → prominence), inflection windows, resample
        and whitened-cosine match — all on x.device. Bit-parity with the former
        NumPy/scipy path (peaks/windows identical; m to ~1e-7).

        Returns:
            emb  : (B, P, C, patch_width) embedding segments
            m    : (B, P, K) whitened-cosine match
            mask : (B, P) bool valid-peak mask
            cen  : (B, P) peak-centre sample indices (float)
            bnd  : (B, P, 2) peak window [lo, hi] (int)
        """
        dev = x.device
        X = x.detach()
        B, C, T = X.shape
        P, pw, K = self.max_peaks, self.patch_width, self.K
        proto_w = self.proto_w
        mind = max(1, int(round(40.0 / 1000.0 * self.sfreq)))

        # Smooth the detection channel; detect ± peaks (distance then prominence).
        s = _t_smooth(X[:, self.detect_ch, :], TRIAL_SMOOTH_SIGMA)
        kp, pp = _t_find_peaks(s, self.peak_prominence, mind)
        kn, pn = _t_find_peaks(-s, self.peak_prominence, mind)

        # Pool ± candidates; keep top-P by prominence with numpy's exact tiebreak
        # (prominence desc → pos-before-neg → lower index), then temporal order.
        candmask = kp | kn
        candprom = torch.where(kp, pp, torch.where(kn, pn, torch.full_like(pp, -1e30)))
        candprom = torch.where(candmask, candprom, torch.full_like(candprom, -1e30))
        polar = torch.where(kp, torch.zeros_like(candprom),
                            torch.where(kn, torch.ones_like(candprom),
                                        torch.full_like(candprom, 2.0)))
        colidx = torch.arange(T, device=dev).expand(B, T)
        perm = colidx.clone()
        _, o = torch.sort(torch.gather(polar, 1, perm), dim=1, stable=True)
        perm = torch.gather(perm, 1, o)
        _, o = torch.sort(torch.gather(-candprom, 1, perm), dim=1, stable=True)
        perm = torch.gather(perm, 1, o)
        keepmask = torch.zeros(B, T, dtype=torch.bool, device=dev)
        keepmask.scatter_(1, perm[:, :P], True)
        keepmask = keepmask & candmask
        keypos = torch.where(keepmask, colidx, torch.full_like(colidx, T + 1))
        pos_sorted, _ = keypos.sort(dim=1)
        pos_sorted = pos_sorted[:, :P]
        mask = pos_sorted <= (T - 1)

        # Inflection-bounded windows (single trials don't cross zero): each peak's
        # bare flanking second-derivative shoulders, ≥2-sample floor.
        d2 = s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]
        chg = torch.sign(d2)[:, 1:] != torch.sign(d2)[:, :-1]
        is_infl = torch.zeros(B, T, dtype=torch.bool, device=dev)
        is_infl[:, 1:T - 2] = chg
        lo_all = _t_nearest(is_infl, True)
        hi_all = _t_nearest(is_infl, False)
        pk = pos_sorted.clamp(0, T - 1)
        lo = torch.where(mask, torch.gather(lo_all, 1, pk), torch.zeros_like(pk))
        hi = torch.where(mask, torch.gather(hi_all, 1, pk), torch.zeros_like(pk))
        short = mask & ((hi - lo) < 2)
        hi = torch.where(short, torch.minimum(torch.full_like(hi, T), lo + 2), hi)
        valid = mask & (hi > lo)
        cen = torch.where(valid, pos_sorted, torch.zeros_like(pos_sorted)).float()
        bnd = torch.stack([lo, hi], dim=-1)

        # Resample every valid unit (routing width pw + each prototype's native
        # width) and form the whitened-cosine match — one matmul per prototype.
        emb = torch.zeros(B, P, C, pw, device=dev)
        m = torch.zeros(B, P, K, device=dev)
        bidx, jidx = torch.where(valid)
        if bidx.numel() > 0:
            los, his = lo[bidx, jidx], hi[bidx, jidx]
            emb[bidx, jidx] = _t_resample(X, bidx, los, his, pw)
            for k in range(K):
                wk = int(proto_w[k].item())
                if wk <= 0:
                    continue
                dk = C * wk
                seg = _t_resample(X, bidx, los, his, wk).reshape(-1, dk)
                V = seg @ self.whitener[k, :dk, :dk].T
                vn = V.norm(dim=1) + 1e-8
                m[bidx, jidx, k] = (V @ self.proto_white[k, :dk]) / (vn * self.proto_white_norm[k])
        return emb, m, valid, cen, bnd

    def _self_attention(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked multi-head self-attention over the trial's peak tokens, with
        residual. Only the actually-detected peaks attend to each other; the
        padded slots are excluded (no attending to empty padding)."""
        B, N, d = z.shape
        H, d_h = self.num_heads, self.d_head
        z_ln = self.sa_ln(z)
        q = self.sa_W_q(z_ln).view(B, N, H, d_h).permute(0, 2, 1, 3)
        k = self.sa_W_k(z_ln).view(B, N, H, d_h).permute(0, 2, 1, 3)
        v = self.sa_W_v(z_ln).view(B, N, H, d_h).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)
        scores = scores.masked_fill((~mask).view(B, 1, 1, N), float('-inf'))
        attn = self.sa_dropout(torch.nan_to_num(F.softmax(scores, dim=-1)))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, d)
        return z + self.sa_out_proj(out)

    def _proto_patch_idx(self) -> torch.Tensor:
        """Prototype PE anchor = window-midpoint PATCH index (matchedcos exactly):
        center_ms = (start_ms+end_ms)/2 → ms_to_sample → // patch_width. The
        ms→sample ROUND happens before the floor-divide (the raw sample midpoint
        would occasionally floor one patch lower).

        Derived from mf_window (a restored buffer), NOT detected_windows_ms (a
        Python attr set only in set_prototypes) — so it survives a checkpoint
        reload without re-running set_prototypes (e.g. the certificate)."""
        idx = []
        for k in range(self.K):
            s = int(self.mf_window[k, 0]); e = int(self.mf_window[k, 1])
            start_ms = round(s / self.sfreq * 1000 + self.tmin * 1000, 1)
            end_ms = round(e / self.sfreq * 1000 + self.tmin * 1000, 1)
            cs = ms_to_sample((start_ms + end_ms) / 2.0, self.sfreq, self.tmin)
            idx.append(max(0, min(cs // self.patch_width, self.n_grid - 1)))
        return torch.tensor(idx, dtype=torch.long, device=self.pos_embed.device)

    def forward(self, x: torch.Tensor):
        """Forward pass → (logit (B,1), aux)."""
        B = x.shape[0]
        H, d_h = self.num_heads, self.d_head

        emb, m, mask, center, bounds = self._tokenize_and_match(x)
        P = emb.shape[1]

        # Peak-token embeddings + PE at the peak index; padded slots are zeroed
        # (masked self-attention and the m·mask readout both exclude them).
        z = self.patch_embed(emb) + self._interp_pos(center)
        z = z * mask.unsqueeze(-1)
        z = self.dropout_layer(z)
        if self.use_self_attn:
            z = self._self_attention(z, mask)

        # Prototype KEY = the compact resampled window segment (proto_seg) patch-
        # embedded — a sharp, non-diluted fingerprint — + PE at the window-midpoint
        # patch. (A full-template mean-pool key was tested and matched within noise
        # but is more diluted; the compact key is the cleaner choice.)
        p = self.patch_embed(self.proto_seg) + self.pos_embed[self._proto_patch_idx()]
        p_exp = p.unsqueeze(0).expand(B, -1, -1)

        # QK cross-attention: peaks query, prototypes key → a[b,p,k].
        q = self.W_q(self.ln_q(z)).view(B, P, H, d_h).permute(0, 2, 1, 3)
        k = self.W_k(self.ln_kv(p_exp)).view(B, self.K, H, d_h).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)
        scores = scores.clamp(-10, 10)
        a = F.softmax(scores, dim=-1).mean(dim=1)          # (B, P, K)

        # Fixed grounded bilinear readout: s · Σ_{p,k} a·m / (#valid peaks) + bias.
        contrib = (a * m) * mask.unsqueeze(-1)
        n_valid = mask.sum(dim=1).clamp_min(1).float()
        logit = self.match_scale * contrib.sum(dim=(1, 2)) / n_valid

        aux = {"a": a, "m": m, "mask": mask, "center": center, "bounds": bounds}
        return logit.unsqueeze(-1) + self.match_bias, aux

    @torch.no_grad()
    def compute_mf(self, x: torch.Tensor) -> torch.Tensor:
        """Amplitude matched-filter factor MF[b,k] = ⟨trial, template_k⟩/‖t_k‖.

        Raw projection of the trial onto each template over its window — the
        template norm is the only normalization (no test-subject/trial stats).
        Returns (B, K).
        """
        B = x.shape[0]
        mf = torch.zeros(B, self.K, device=x.device)
        for k in range(self.K):
            s, e = int(self.mf_window[k, 0]), int(self.mf_window[k, 1])
            if e <= s:
                continue
            tmpl = self.mf_template[k, :, s:e]
            proj = (x[:, :, s:e] * tmpl.unsqueeze(0)).sum(dim=(1, 2))
            mf[:, k] = proj / self.mf_template_norm[k]
        return mf

    @torch.no_grad()
    def compute_mf_channel(self, x: torch.Tensor):
        """Channel-resolved amplitude matched filter + inter-channel contrasts.

        Spatial decomposition of the scalar MF: instead of collapsing channels,
        keep the per-channel projection onto each prototype template, and the
        channel-DIFFERENCE (bipolar) projection — capturing the spatial gradient
        (e.g. Pz−Fz for the P3b, PO7−PO8 for N2pc) that the scalar MF hides. All
        grounded: fixed template over its window, template-norm only, no
        test-subject/trial stats. MF_kc uses the GLOBAL template norm so
        Σ_c MF_kc = MF_k (a true decomposition). Certified component-specific
        (on-window ≫ off-window).

        Returns (MF_kc (B, K, C), contrast (B, K, C·(C−1)/2)).
        """
        B, C = x.shape[0], self.n_channels
        pairs = channel_pairs(C)
        mfc = torch.zeros(B, self.K, C, device=x.device)
        ctr = torch.zeros(B, self.K, len(pairs), device=x.device)
        for k in range(self.K):
            s, e = int(self.mf_window[k, 0]), int(self.mf_window[k, 1])
            if e <= s:
                continue
            tk = self.mf_template[k, :, s:e]                 # (C, w)
            seg = x[:, :, s:e]                               # (B, C, w)
            gn = self.mf_template_norm[k]
            mfc[:, k, :] = (seg * tk.unsqueeze(0)).sum(dim=2) / gn
            for pi, (i, j) in enumerate(pairs):
                tc = tk[i] - tk[j]
                ctr[:, k, pi] = ((seg[:, i] - seg[:, j]) * tc.unsqueeze(0)).sum(dim=1) / gn
        return mfc, ctr


def fusion_feature_metadata(model: "ERPXTTN") -> dict:
    """Human/audit metadata for the Stage-2 fusion feature vector."""
    channels = [str(c) for c in getattr(
        model, "channel_names", [f"ch{i}" for i in range(model.n_channels)])]
    pairs = channel_pairs(model.n_channels)

    names = ["routing_logit"]
    slices = {"routing_logit": [0, 1]}
    start = 1

    names.extend([f"MF_proto{k}" for k in range(model.K)])
    slices["mf"] = [start, start + model.K]
    start += model.K

    names.extend([
        f"MF_proto{k}_channel_{channels[c]}"
        for k in range(model.K) for c in range(model.n_channels)
    ])
    slices["mf_channel"] = [start, start + model.K * model.n_channels]
    start += model.K * model.n_channels

    names.extend([
        f"MF_proto{k}_contrast_{channels[i]}-{channels[j]}"
        for k in range(model.K) for i, j in pairs
    ])
    slices["mf_contrast"] = [start, start + model.K * len(pairs)]

    return {
        "feature_names": names,
        "feature_slices": slices,
        "n_features": len(names),
        "n_prototypes": int(model.K),
        "n_channels": int(model.n_channels),
        "channel_names": channels,
        "contrast_pairs": [
            {"i": int(i), "j": int(j), "name": f"{channels[i]}-{channels[j]}"}
            for i, j in pairs
        ],
        "feature_order": ["routing_logit", "mf", "mf_channel", "mf_contrast"],
        "normalization": {
            "mf": "global_template_norm",
            "mf_channel": "global_template_norm; sums over channels to MF",
            "mf_contrast": "global_template_norm",
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Two-factor fusion (Stage 2) — the ERP-XTTN headline decision.
#
# The model above emits grounded factors per trial: forward() -> the routing
# logit, compute_mf() -> scalar matched filters, compute_mf_channel() ->
# channel-resolved and bipolar contrast matched filters. Stage 2 fuses them
# across subjects with a per-fold, zero-calibration LOSO logistic: for held-out
# subject s, the frozen fold model f_s (trained without s) computes features for
# the TRAINING subjects, a logistic is fit on those, and s is scored with f_s's
# own features. No other fold's model and no test-subject data enter s's fusion.
# ──────────────────────────────────────────────────────────────────────

def load_frozen_model(ckpt: dict, device) -> "ERPXTTN":
    """Rebuild a frozen ERPXTTN from a checkpoint dict (K read off state_dict)."""
    c = ckpt["model_config"]
    sd = ckpt["state_dict"]
    K = int(sd["proto_seg"].shape[0])
    model = ERPXTTN(
        c["n_channels"], c["n_times"], channel_names=c.get("channel_names"),
        detection_channel=c.get("detection_channel"),
        d_model=c.get("d_model", 64), num_heads=c.get("num_heads", 4),
        patch_width=c.get("patch_width", 8), max_k=K, max_peaks=c["max_peaks"],
        use_self_attn=c["use_self_attn"], sfreq=c.get("sfreq", 256.0),
        peak_prominence=c.get("peak_prominence", 0.02))
    model.load_state_dict(sd)
    return model.eval().to(device)


@torch.no_grad()
def _fold_features(model: "ERPXTTN", Xn: np.ndarray, device, bs: int = 256):
    """Grounded two-factor fusion features per trial (N, D):
    [routing_logit, MF_k, MF_kc (channel-resolved), contrast_MF (bipolar)].
    All from the frozen model over normalized epochs."""
    feats = []
    for i in range(0, len(Xn), bs):
        xb = torch.from_numpy(Xn[i:i + bs]).to(device)
        B = xb.shape[0]
        logit, _ = model(xb)
        mf = model.compute_mf(xb)
        kc, ct = model.compute_mf_channel(xb)
        feats.append(np.column_stack([
            logit.squeeze(-1).cpu().numpy(), mf.cpu().numpy(),
            kc.reshape(B, -1).cpu().numpy(), ct.reshape(B, -1).cpu().numpy()]))
    return np.concatenate(feats)


def two_factor_fusion(results_dir, all_data: dict, device):
    """Per-fold zero-calibration two-factor fusion (routing logit + amplitude MF).

    all_data: {subject: (X_raw (N,C,T), y (N,))} — RAW epochs; each fold
    normalizes with its own checkpoint's (norm_mean, norm_std). Writes
    two_factor_{s}.npz per subject and returns {subject: auroc} (None if any
    fold checkpoint is missing).
    """
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    results_dir = Path(results_dir)
    subjects = list(all_data.keys())
    per_subject = {}
    for s in subjects:
        ckpt_path = results_dir / f"checkpoint_{s}.pt"
        if not ckpt_path.exists():
            return None
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        f_s = load_frozen_model(ckpt, device)
        meta = fusion_feature_metadata(f_s)
        nm, ns = ckpt["norm_mean"], ckpt["norm_std"]

        # Combiner training set = the OTHER subjects through f_s (consistent K_s).
        Xtr, ytr = [], []
        for o in subjects:
            if o == s:
                continue
            Xo, yo = all_data[o]
            Xtr.append(_fold_features(f_s, ((Xo - nm) / ns).astype(np.float32), device))
            ytr.append(yo)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(np.concatenate(Xtr), np.concatenate(ytr))

        # Held-out subject s, same f_s.
        Xs, ys = all_data[s]
        feat_s = _fold_features(f_s, ((Xs - nm) / ns).astype(np.float32), device)
        prob = clf.predict_proba(feat_s)[:, 1]
        auc = float(roc_auc_score(ys, prob))
        per_subject[s] = auc
        np.savez_compressed(str(results_dir / f"two_factor_{s}.npz"),
                            probs=prob, labels=ys, auroc=auc,
                            coef=clf.coef_[0], intercept=clf.intercept_,
                            feature_names=np.asarray(meta["feature_names"], dtype=str),
                            feature_metadata_json=json.dumps(meta),
                            feature_slices_json=json.dumps(meta["feature_slices"]),
                            combiner_features="routing_mf_channel_contrast_v1",
                            lr_class_weight="balanced",
                            lr_penalty="l2")
    return per_subject
