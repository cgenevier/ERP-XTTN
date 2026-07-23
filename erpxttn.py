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

  (2) amplitude / matched-filter — a per-component *raw* projection of the trial
      onto each template (a K-vector per trial), exposed by compute_mf(). It is
      fused with the routing logit OUTSIDE the model, via leave-one-subject-out
      logistic regression over [routing_logit, MF_1..K] (two-stage late fusion).

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

        Returns:
            emb  : (B, P, C, patch_width) embedding segments
            m    : (B, P, K) whitened-cosine match
            mask : (B, P) bool valid-peak mask
            cen  : (B, P) peak-centre sample indices (float)
            bnd  : (B, P, 2) peak window [lo, hi] (int)
        """
        Xnp = x.detach().cpu().numpy()
        B, C, T = Xnp.shape
        P, pw, K = self.max_peaks, self.patch_width, self.K

        emb = np.zeros((B, P, C, pw), dtype=np.float32)
        mask = np.zeros((B, P), dtype=bool)
        cen = np.zeros((B, P), dtype=np.float32)
        bnd = np.zeros((B, P, 2), dtype=np.int64)

        proto_w = self.proto_w.cpu().numpy()
        whit = self.whitener.cpu().numpy()
        pwhite = self.proto_white.cpu().numpy()
        pwnorm = self.proto_white_norm.cpu().numpy()

        # Batch-smooth the detection channel ONCE (matchedcos smooths the whole
        # batch at once) — canonical sigma=3.0 for trials (prototypes stay 2.0).
        smoothed_all = _smooth_trial_batch(Xnp[:, self.detect_ch], TRIAL_SMOOTH_SIGMA)

        # find_peaks is per-trial (unavoidable); collect every valid unit's
        # window here, then resample + match ALL units at once (vectorized).
        bs, js, los, his = [], [], [], []
        for b in range(B):
            sig = Xnp[b, self.detect_ch]
            smoothed = smoothed_all[b]
            peaks = detect_trial_peaks(
                sig, self.sfreq, prominence=self.peak_prominence, max_peaks=P,
                presmoothed=smoothed)
            if not peaks:
                continue
            idx = [p[0] for p in peaks]
            # Trial peaks are inflection-bounded (single trials don't cross zero),
            # using bare flanking inflections (no width clamp / no-overlap).
            wins = build_windows_from_inflections(idx, smoothed, self.sfreq)
            for j, (lo, hi) in enumerate(wins):
                if hi <= lo:
                    continue
                mask[b, j] = True
                # PE anchor = the PEAK INDEX (canonical: matchedcos centers[b,:n]=keep),
                # NOT the window midpoint — the peak is where the deflection actually is.
                cen[b, j] = idx[j]
                bnd[b, j] = (lo, hi)
                bs.append(b); js.append(j); los.append(lo); his.append(hi)

        # Vectorized native-width whitened-cosine match, one matmul per prototype.
        m = np.zeros((B, P, K), dtype=np.float32)
        if bs:
            bs = np.asarray(bs, dtype=np.int64)
            js = np.asarray(js, dtype=np.int64)
            los = np.asarray(los, dtype=np.int64)
            his = np.asarray(his, dtype=np.int64)

            # Routing embedding: width-pw resample of every valid unit at once
            # (padded slots stay zero and are masked out downstream).
            emb[bs, js] = _resample_batch(Xnp, bs, los, his, pw)

            for k in range(K):
                wk = int(proto_w[k])
                if wk <= 0:
                    continue
                dk = C * wk
                seg = _resample_batch(Xnp, bs, los, his, wk).reshape(len(bs), dk)
                V = seg @ whit[k, :dk, :dk].T               # (N, dk)
                vn = np.linalg.norm(V, axis=1) + 1e-8
                m[bs, js, k] = (V @ pwhite[k, :dk]) / (vn * pwnorm[k])
        m *= mask[:, :, None]

        dev = x.device
        return (torch.from_numpy(emb).to(dev), torch.from_numpy(m).to(dev),
                torch.from_numpy(mask).to(dev), torch.from_numpy(cen).to(dev),
                torch.from_numpy(bnd).to(dev))

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
