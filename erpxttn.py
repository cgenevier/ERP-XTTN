"""ERP-XTTN — cross-attention ErrP classifier with dynamic prototypes,
self-attention, and prototype positional encoding.

Classifies EEG epochs by cross-attending input patches against ERP prototype
templates derived from the grand-average difference wave. Prototype temporal
windows are detected per fold via alternating-polarity peak finding. A
self-attention layer refines patch embeddings before cross-attention, and
prototypes receive shared positional encoding for temporal alignment.

Architecture:
    1. Patch embed + positional encoding  ->  z  (B, N, d)
    2. Self-attention: LN -> MHSA -> residual  ->  z'  (B, N, d)
    3. Prototype embed + positional encoding (shared PE)  ->  p  (K, d)
    4. Cross-attention (QK-only) with prototypes  ->  attention weights
    5. Classification from attention routing
"""

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Canonical detection channel for ERP components
DETECT_CHANNEL = "Cz"

# Minimum latency (ms) for P1 anchor — prevents locking onto early
# noise/artifact before the physiological ERP response.
MIN_P1_LATENCY_MS = 50.0


def ms_to_sample(ms: float, sfreq: float = 256.0, tmin: float = 0.0) -> int:
    """Convert milliseconds to sample index."""
    return int(round((ms / 1000.0 - tmin) * sfreq))


# ──────────────────────────────────────────────────────────────────────
# Dynamic peak detection
# ──────────────────────────────────────────────────────────────────────

def _smooth_signal(diff_signal: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth a 1-D signal for robust peak/zero-crossing detection."""
    return gaussian_filter1d(diff_signal, sigma=sigma)


def _find_zero_crossings(signal: np.ndarray) -> np.ndarray:
    """Find sample indices where the signal crosses zero.

    Returns indices of the last sample before each sign change.
    """
    signs = np.sign(signal)
    for i in range(len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1] if i > 0 else 1
    changes = np.where(np.diff(signs) != 0)[0]
    return changes


def detect_alternating_peaks(
    diff_signal: np.ndarray, sfreq: float,
    polarity_pattern: list[str] = None,
    prominence: float = 0.1,
    min_distance_ms: float = 31.0,
    smooth_sigma: float = 2.0,
    min_latency_ms: float = 50.0,
) -> list[tuple[int, str]]:
    """Detect ERP component peaks enforcing alternating P-N-P-N polarity.

    The ErrP difference wave has a characteristic positive-negative-positive-
    negative pattern (P1 -> Ne -> Pe -> LateN). This function finds the
    temporally-ordered chain of peaks with that polarity sequence that
    maximizes total prominence.

    Args:
        diff_signal: (T,) difference wave on the detection channel (z-scored)
        sfreq: sampling frequency in Hz
        polarity_pattern: expected polarity sequence, e.g. ['pos','neg','pos','neg']
        prominence: minimum peak prominence (in z-score units)
        min_distance_ms: minimum inter-peak distance in ms
        smooth_sigma: Gaussian smoothing sigma in samples
        min_latency_ms: minimum latency for P1 anchor (ms post-stimulus)

    Returns:
        List of (sample_index, polarity) tuples sorted by time,
        where polarity is 'pos' or 'neg'.
    """
    if polarity_pattern is None:
        polarity_pattern = ['pos', 'neg', 'pos', 'neg']

    min_distance = max(1, int(round(min_distance_ms / 1000.0 * sfreq)))
    min_latency_sample = int(round(min_latency_ms / 1000.0 * sfreq))
    smoothed = _smooth_signal(diff_signal, sigma=smooth_sigma)

    # Find all positive peaks
    pos_peaks, pos_props = find_peaks(
        smoothed, prominence=prominence, distance=min_distance)
    pos_list = sorted([(int(idx), float(prom)) for idx, prom in
                       zip(pos_peaks, pos_props['prominences'])])

    # Find all negative peaks (peaks of inverted signal)
    neg_peaks, neg_props = find_peaks(
        -smoothed, prominence=prominence, distance=min_distance)
    neg_list = sorted([(int(idx), float(prom)) for idx, prom in
                       zip(neg_peaks, neg_props['prominences'])])

    # Build candidate lists per slot in the pattern
    slot_candidates = []
    for polarity in polarity_pattern:
        slot_candidates.append(pos_list if polarity == 'pos' else neg_list)

    # Anchor on the earliest peak (of the first slot's polarity) at or
    # after min_latency_ms, then find the best chain via recursive search.
    first_candidates = [
        (idx, prom) for idx, prom in slot_candidates[0]
        if idx >= min_latency_sample
    ]
    if not first_candidates:
        return []

    # Pick the earliest peak for the first slot
    p1_idx, p1_prom = first_candidates[0]

    # Find best chain for remaining slots after P1
    best_chain: list[tuple[int, float]] = []
    best_score = -1.0

    def search(slot: int, min_sample: int, chain: list, score: float):
        nonlocal best_chain, best_score

        if slot == len(polarity_pattern):
            if score > best_score:
                best_score = score
                best_chain = list(chain)
            return

        for idx, prom in slot_candidates[slot]:
            if idx < min_sample:
                continue
            chain.append((idx, prom))
            search(slot + 1, idx + min_distance, chain, score + prom)
            chain.pop()

    search(1, p1_idx + min_distance, [(p1_idx, p1_prom)], p1_prom)

    if not best_chain:
        return []

    return [(idx, polarity_pattern[i]) for i, (idx, _) in enumerate(best_chain)]


def build_windows_from_zero_crossings(
    peak_indices: list[int], smoothed_signal: np.ndarray,
    min_window_ms: float = 40.0,
    max_window_ms: float = 200.0,
    sfreq: float = 256.0,
) -> list[tuple[int, int]]:
    """Build windows by expanding from peaks to neighboring zero-crossings.

    Each window spans from the zero-crossing before the peak to the zero-
    crossing after it, giving the natural width of each ERP deflection.

    Args:
        peak_indices: sorted list of peak sample indices
        smoothed_signal: (T,) smoothed difference wave used for zero-crossing
        min_window_ms: minimum window width in ms
        max_window_ms: maximum window width in ms
        sfreq: sampling frequency

    Returns:
        List of (start_sample, end_sample) tuples.
    """
    T = len(smoothed_signal)
    min_w = int(round(min_window_ms / 1000.0 * sfreq))
    max_w = int(round(max_window_ms / 1000.0 * sfreq))

    crossings = _find_zero_crossings(smoothed_signal)

    windows = []
    for i, peak in enumerate(peak_indices):
        # Find the zero-crossing just before the peak
        before = crossings[crossings < peak]
        left = int(before[-1]) + 1 if len(before) > 0 else 0

        # Find the zero-crossing just after the peak
        after = crossings[crossings >= peak]
        right = int(after[0]) + 1 if len(after) > 0 else T

        # Clamp to not overlap with previous window
        if i > 0:
            prev_right = windows[-1][1]
            left = max(left, prev_right)

        # Enforce minimum width (expand symmetrically if needed)
        if right - left < min_w:
            deficit = min_w - (right - left)
            expand_left = deficit // 2
            expand_right = deficit - expand_left
            left = max(left - expand_left, 0 if i == 0 else windows[-1][1])
            right = min(right + expand_right, T)

        # Enforce maximum width (shrink symmetrically)
        if right - left > max_w:
            excess = (right - left - max_w) // 2
            left += excess
            right = left + max_w

        # Final boundary clamp
        left = max(left, 0)
        right = min(right, T)

        windows.append((left, right))

    return windows


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split (B, C, T) into non-overlapping patches and project."""

    def __init__(self, n_channels: int, patch_width: int, d_model: int):
        super().__init__()
        self.patch_width = patch_width
        self.proj = nn.Linear(n_channels * patch_width, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        w = self.patch_width
        N = T // w
        x = x[:, :, :N * w].reshape(B, C, N, w).permute(0, 2, 1, 3).reshape(B, N, C * w)
        return self.proj(x)


class ERPXTTN(nn.Module):
    """ERP-XTTN: cross-attention ErrP classifier.

    Input:  (B, C, T)
    Output: (B, 1) logit

    Call set_prototypes() before the first forward pass in each fold.
    """

    def __init__(self, n_channels: int, n_times: int,
                 channel_names: list[str] = None,
                 d_model: int = 64, num_heads: int = 4,
                 patch_width: int = 8, dropout: float = 0.3,
                 sfreq: float = 256.0, tmin: float = 0.0,
                 n_proto: int = 4,
                 polarity_pattern: list[str] = None,
                 peak_prominence: float = 0.1,
                 min_window_ms: float = 40.0,
                 max_window_ms: float = 200.0,
                 routing_contrast_weight: float = 0.0,
                 detection_channel: str = None,
                 peak_mode: str = 'constrained',
                 max_k: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.patch_width = patch_width
        self.K = n_proto
        self.polarity_pattern = polarity_pattern or ['pos', 'neg', 'pos', 'neg']
        self.peak_prominence = peak_prominence
        self.peak_mode = peak_mode
        self.max_k = max_k
        self.N = n_times // patch_width
        self.sfreq = sfreq
        self.tmin = tmin
        self.routing_contrast_weight = routing_contrast_weight

        # Resolve detection channel by name
        detect_name = detection_channel or DETECT_CHANNEL
        if channel_names is not None and detect_name in channel_names:
            self.detect_ch = channel_names.index(detect_name)
        else:
            self.detect_ch = min(1, n_channels - 1)
            if channel_names is not None:
                logging.warning(
                    f"Detection channel '{detect_name}' not found in "
                    f"{channel_names}; falling back to index {self.detect_ch}")

        # Dynamic window parameters
        self.min_window_ms = min_window_ms
        self.max_window_ms = max_window_ms

        # Detected windows (set by set_prototypes, used for saving metadata)
        self.detected_windows_ms: list[tuple[float, float]] = []

        # Patch embedding (shared for input and prototypes)
        self.patch_embed = PatchEmbedding(n_channels, patch_width, d_model)

        # Learned positional embeddings for input patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.N, d_model) * 0.02)

        # Cross-attention: separate layer norms for Q and K
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)

        # Q and K projections only (no V)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)

        # Classification from attention weights
        self.head = nn.Linear(self.N * self.K, 1)

        self.dropout_layer = nn.Dropout(dropout)

        # Prototype buffer — set per fold via set_prototypes()
        self.register_buffer("proto_raw",
                             torch.zeros(self.K, n_channels, n_times))

        # Self-attention components
        self.sa_ln = nn.LayerNorm(d_model)
        self.sa_W_q = nn.Linear(d_model, d_model)
        self.sa_W_k = nn.Linear(d_model, d_model)
        self.sa_W_v = nn.Linear(d_model, d_model)
        self.sa_out_proj = nn.Linear(d_model, d_model)
        self.sa_dropout = nn.Dropout(dropout)

        # Prototype center patch indices for PE lookup (set per fold)
        self.register_buffer("proto_center_patch_idx",
                             torch.zeros(self.K, dtype=torch.long))

    def _auto_detect_peaks(self, diff_signal, sfreq):
        """Auto-detect prominent peaks without a polarity pattern.

        Finds all positive and negative peaks above the prominence threshold,
        enforces sign constraint (pos peaks > 0, neg peaks < 0), ranks by
        prominence, and takes top max_k. Returns list of (sample_idx, polarity).
        """
        from scipy.signal import find_peaks as _find_peaks

        AUTO_PROMINENCE = 0.02
        min_distance = max(1, int(round(80.0 / 1000.0 * sfreq)))
        min_latency_sample = int(round(MIN_P1_LATENCY_MS / 1000.0 * sfreq))
        smoothed = _smooth_signal(diff_signal, sigma=2.0)

        pos_peaks, pos_props = _find_peaks(smoothed, prominence=AUTO_PROMINENCE,
                                            distance=min_distance)
        neg_peaks, neg_props = _find_peaks(-smoothed, prominence=AUTO_PROMINENCE,
                                            distance=min_distance)

        all_peaks = []
        for idx, prom in zip(pos_peaks, pos_props['prominences']):
            if idx >= min_latency_sample and smoothed[idx] > 0:
                all_peaks.append((int(idx), 'pos', float(prom)))
        for idx, prom in zip(neg_peaks, neg_props['prominences']):
            if idx >= min_latency_sample and smoothed[idx] < 0:
                all_peaks.append((int(idx), 'neg', float(prom)))

        all_peaks.sort(key=lambda x: x[2], reverse=True)
        selected = all_peaks[:self.max_k]
        selected.sort(key=lambda x: x[0])

        return [(idx, pol) for idx, pol, _ in selected]

    def set_prototypes(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Detect ERP peaks and compute diff-wave prototypes.

        Supports two modes:
        - 'constrained': enforces polarity pattern chain (original method)
        - 'auto': data-driven, finds top-K peaks by prominence

        Args:
            X_train: (N_train, C, T) normalized training epochs
            y_train: (N_train,) labels (0=correct, 1=error)
        """
        error_avg = X_train[y_train == 1].mean(dim=0)    # (C, T)
        correct_avg = X_train[y_train == 0].mean(dim=0)  # (C, T)
        diff_wave = error_avg - correct_avg               # (C, T)

        # Extract detection channel signal (ensure CPU for scipy)
        diff_signal = diff_wave[self.detect_ch].cpu().numpy()
        smoothed = _smooth_signal(diff_signal, sigma=2.0)

        if self.peak_mode == 'auto':
            peaks = self._auto_detect_peaks(diff_signal, self.sfreq)
            if len(peaks) == 0:
                raise RuntimeError(
                    f"Auto peak detection found no peaks above prominence "
                    f"{self.peak_prominence}. Check preprocessing and data quality.")
            # Update K to match actual number of peaks found
            actual_k = len(peaks)
            if actual_k != self.K:
                logging.info(f"  Auto-detect: adjusting K from {self.K} to {actual_k}")
                self.K = actual_k
                # Resize buffers
                self.proto_raw = nn.Parameter(
                    torch.zeros(self.K, self.n_channels, self.n_times),
                    requires_grad=False)
                self.proto_center_patch_idx = nn.Parameter(
                    torch.zeros(self.K, dtype=torch.long),
                    requires_grad=False)
                # Resize head: input is N_patches * K
                self.head = nn.Linear(self.N * self.K, 1)
                self.to(next(self.parameters()).device)
        else:
            peaks = detect_alternating_peaks(
                diff_signal, self.sfreq,
                polarity_pattern=self.polarity_pattern,
                prominence=self.peak_prominence,
                min_latency_ms=MIN_P1_LATENCY_MS,
            )
            if len(peaks) < self.K:
                raise RuntimeError(
                    f"Dynamic peak detection found only {len(peaks)}/{self.K} "
                    f"peaks on the grand-average difference wave. Expected "
                    f"polarity pattern {self.polarity_pattern}. Check "
                    f"preprocessing and data quality. Detected peaks: {peaks}")

        peak_indices = [p[0] for p in peaks]

        # Build windows from zero-crossings
        raw_windows = build_windows_from_zero_crossings(
            peak_indices, smoothed,
            min_window_ms=self.min_window_ms,
            max_window_ms=self.max_window_ms,
            sfreq=self.sfreq,
        )

        # In auto mode, drop peaks whose windows don't contain them
        if self.peak_mode == 'auto':
            valid = [(p, w) for p, w in zip(peaks, raw_windows)
                     if w[0] <= p[0] <= w[1]]
            if len(valid) < len(peaks):
                dropped = len(peaks) - len(valid)
                logging.info(f"  Auto-detect: dropped {dropped} peak(s) with invalid windows")
                peaks = [v[0] for v in valid]
                raw_windows = [v[1] for v in valid]
                actual_k = len(peaks)
                if actual_k != self.K:
                    self.K = actual_k
                    self.proto_raw = nn.Parameter(
                        torch.zeros(self.K, self.n_channels, self.n_times),
                        requires_grad=False)
                    self.proto_center_patch_idx = nn.Parameter(
                        torch.zeros(self.K, dtype=torch.long),
                        requires_grad=False)
                    self.head = nn.Linear(self.N * self.K, 1)
                    self.to(next(self.parameters()).device)

        windows_samples = raw_windows
        self.detected_windows_ms = [
            (round(s / self.sfreq * 1000 + self.tmin * 1000, 1),
             round(e / self.sfreq * 1000 + self.tmin * 1000, 1))
            for s, e in windows_samples
        ]

        # Extract diff-wave within detected windows
        proto = torch.zeros(self.K, self.n_channels, self.n_times)
        for k, (s, e) in enumerate(windows_samples):
            proto[k, :, s:e] = diff_wave[:, s:e].cpu()

        self.proto_raw.copy_(proto)

        # Compute center patch index for each prototype for PE lookup
        indices = []
        for start_ms, end_ms in self.detected_windows_ms:
            center_ms = (start_ms + end_ms) / 2.0
            center_sample = ms_to_sample(center_ms, self.sfreq, self.tmin)
            center_patch_idx = center_sample // self.patch_width
            center_patch_idx = max(0, min(center_patch_idx, self.N - 1))
            indices.append(center_patch_idx)

        self.proto_center_patch_idx.copy_(
            torch.tensor(indices, dtype=torch.long))

    def _embed_prototypes(self) -> torch.Tensor:
        """Embed raw prototypes through patch embedding and mean-pool.

        Returns: (K, d_model)
        """
        z = self.patch_embed(self.proto_raw)   # (K, N, d_model)
        return z.mean(dim=1)                   # (K, d_model)

    def _self_attention(self, z: torch.Tensor) -> torch.Tensor:
        """Apply one layer of multi-head self-attention with residual.

        Args:
            z: (B, N, d) patch embeddings

        Returns:
            z': (B, N, d) refined patch embeddings
        """
        B, N, d = z.shape
        H, d_h = self.num_heads, self.d_head

        z_ln = self.sa_ln(z)

        q = self.sa_W_q(z_ln).view(B, N, H, d_h).permute(0, 2, 1, 3)
        k = self.sa_W_k(z_ln).view(B, N, H, d_h).permute(0, 2, 1, 3)
        v = self.sa_W_v(z_ln).view(B, N, H, d_h).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)
        attn = F.softmax(scores, dim=-1)
        attn = self.sa_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, d)
        out = self.sa_out_proj(out)

        return z + out

    def _compute_proto_features(self, attn_avg: torch.Tensor) -> torch.Tensor:
        """Compute prototype routing summary features from attention map.

        These features characterize how the trial's attention relates to the
        prototypes: how much each prototype is attended, how peaked/diffuse
        the attention is, whether it follows a diagonal temporal structure,
        and where each prototype's attention concentrates in time.

        Args:
            attn_avg: (B, N, K) head-averaged attention map

        Returns:
            (B, 2K+3) feature vector:
                proto_mass (K) — total attention per prototype
                proto_confidence (1) — mean max-attention-per-patch
                proto_entropy (1) — mean per-patch entropy over prototypes
                proto_diagonality (1) — attention to nearest prototype
                proto_com (K) — temporal center of mass per prototype
        """
        B, N, K = attn_avg.shape

        # Per-prototype mass: total attention each prototype receives
        proto_mass = attn_avg.sum(dim=1)  # (B, K)

        # Mean confidence: average of max-attention-per-patch
        proto_confidence = attn_avg.max(dim=-1).values.mean(dim=-1, keepdim=True)  # (B, 1)

        # Mean entropy: average per-patch entropy over prototype dimension
        proto_entropy = -(attn_avg * (attn_avg + 1e-8).log()).sum(dim=-1).mean(dim=-1, keepdim=True)  # (B, 1)

        # Diagonality: attention to nearest prototype by temporal center
        proto_centers = self.proto_center_patch_idx.float()  # (K,)
        patch_indices = torch.arange(N, device=attn_avg.device, dtype=torch.float32)
        dist = (patch_indices.unsqueeze(1) - proto_centers.unsqueeze(0)).abs()  # (N, K)
        nearest_proto = dist.argmin(dim=-1)  # (N,)
        nearest_expanded = nearest_proto.unsqueeze(0).expand(B, -1)  # (B, N)
        diag_attn = attn_avg.gather(dim=-1, index=nearest_expanded.unsqueeze(-1)).squeeze(-1)
        proto_diagonality = diag_attn.mean(dim=-1, keepdim=True)  # (B, 1)

        # Per-prototype temporal center of mass
        patch_pos = torch.arange(N, device=attn_avg.device, dtype=torch.float32)
        attn_t = attn_avg.transpose(1, 2)  # (B, K, N)
        proto_com = (attn_t * patch_pos.unsqueeze(0).unsqueeze(0)).sum(dim=-1) / \
                    (attn_t.sum(dim=-1) + 1e-8)  # (B, K)

        return torch.cat([proto_mass, proto_confidence, proto_entropy,
                          proto_diagonality, proto_com], dim=-1)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Returns (depends on routing_contrast_weight):
            If routing_contrast_weight == 0 (standard mode):
                (logit (B, 1), attn_weights (B, H, N, K))
            If routing_contrast_weight > 0 (RCL mode):
                (logit (B, 1), proto_features (B, 2K+3), attn_weights (B, H, N, K))
        """
        B = x.shape[0]
        H, d_h = self.num_heads, self.d_head

        # --- Patch embedding + positional ---
        z = self.patch_embed(x)        # (B, N, d)
        z = z + self.pos_embed         # (B, N, d)
        z = self.dropout_layer(z)

        # --- Self-attention (pre-cross-attention) ---
        z = self._self_attention(z)    # (B, N, d)

        # --- Prototype embedding + positional encoding ---
        p = self._embed_prototypes()   # (K, d)
        proto_pe = self.pos_embed[0, self.proto_center_patch_idx, :]  # (K, d)
        p = p + proto_pe

        # --- Cross-attention (QK only) ---
        q = self.W_q(self.ln_q(z))                                    # (B, N, d)
        k = self.W_k(self.ln_kv(p.unsqueeze(0).expand(B, -1, -1)))   # (B, K, d)

        # Reshape to multi-head
        q = q.view(B, self.N, H, d_h).permute(0, 2, 1, 3)     # (B, H, N, d_h)
        k = k.view(B, self.K, H, d_h).permute(0, 2, 1, 3)     # (B, H, K, d_h)

        # Scaled dot-product
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)
        logits = logits.clamp(-10, 10)
        attn_weights = F.softmax(logits, dim=-1)    # (B, H, N, K)

        # --- Classification from attention ---
        attn_avg = attn_weights.mean(dim=1)         # (B, N, K)
        attn_flat = attn_avg.reshape(B, -1)         # (B, N*K)
        out = self.head(attn_flat)                   # (B, 1)

        if self.routing_contrast_weight > 0:
            proto_feat = self._compute_proto_features(attn_avg)
            return out, proto_feat, attn_weights

        return out, attn_weights
