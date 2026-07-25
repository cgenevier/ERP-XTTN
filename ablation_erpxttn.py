"""ablation_erpxttn.py — ablation-only variants of ERP-XTTN, ISOLATED from the
production model so `erpxttn.py` (the faithful model the paper presents) stays
untouched. A thin subclass of ERPXTTN; only the differing pieces are overridden.

Modes (exactly one per run, selected by `ablation_mode`):

  nowhiten        Raw-cosine match: the joint whitener Σ_k^(−1/2) is replaced by the
                  identity, so m[p,k] = cosine(raw peak window, raw template) with no
                  whitening. Everything else (routing readout, two-factor fusion)
                  unchanged → tests §2.3.3's whitened-match argument. Fully
                  ERPXTTN-buffer-compatible, so the standard Stage-2 fusion and
                  load_frozen_model work as-is.

  e2e             End-to-end joint head: instead of the grounded readout + a frozen
                  post-hoc LOSO combiner, a single learnable linear head over
                  [routing_logit, MF_kc, contrast] is trained by BCE with the rest of
                  the model. This IS the fusion, trained jointly (no Stage-2). Tests
                  §2.3.5's claim that late fusion is deliberate. 04_train skips the
                  post-hoc combiner for this mode; compare its AUROC to base fusion
                  AND to routing-only.

  learned_readout Free head: resurrects a learned readout over the attention. A value
                  projection is applied to the prototype keys, the peaks attend
                  (Σ_k a[p,k]·V_k), the result is masked-mean-pooled over peaks and a
                  linear head produces the logit. The grounded match m does NOT enter
                  the decision → the "grounded costs ~0 AUROC but the free head fails
                  the certificate" contrast. 04_train reports its forward AUROC
                  (routing-path) and skips the Stage-2 combiner.

The aux dict (a, m, mask, center, bounds) is still returned in every mode, so the
routing dumps and figures keep working.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from erpxttn import ERPXTTN, channel_pairs

ABLATION_MODES = ("nowhiten", "e2e", "learned_readout")


class ERPXTTNAblation(ERPXTTN):
    def __init__(self, *args, ablation_mode: str = "nowhiten", **kwargs):
        super().__init__(*args, **kwargs)
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(f"ablation_mode must be one of {ABLATION_MODES}, got {ablation_mode}")
        self.ablation_mode = ablation_mode

        # The learned_readout / e2e heads need K, fixed only after set_prototypes();
        # they are created there so they are present before the optimizer is built
        # in 04_train (which calls set_prototypes, then AdamW(model.parameters())).

    # ── nowhiten (identity whitener) / e2e head creation happen at prototype time ──
    def set_prototypes(self, X_train: torch.Tensor, y_train: torch.Tensor):
        super().set_prototypes(X_train, y_train)  # builds windows/templates/whiteners as usual
        dev = self.match_scale.device

        if self.ablation_mode == "nowhiten":
            # Overwrite the whitener with identity and the whitened template with
            # the RAW template, so the match becomes a plain cosine. The buffers
            # keep their ERPXTTN shapes, so reload/fusion are unaffected.
            for k in range(self.K):
                wk = int(self.proto_w[k].item())
                if wk <= 0:
                    continue
                d = self.n_channels * wk
                W = torch.zeros_like(self.whitener[k])
                idx = torch.arange(d, device=self.whitener.device)
                W[idx, idx] = 1.0
                self.whitener[k] = W
                s, e = int(self.mf_window[k, 0]), int(self.mf_window[k, 1])
                raw = self.mf_template[k, :, s:e].reshape(-1)      # (d,), channel-major
                self.proto_white[k].zero_()
                self.proto_white[k, :d] = raw
                self.proto_white_norm[k] = float(raw.norm().item() + 1e-8)

        if self.ablation_mode == "e2e":
            npairs = len(channel_pairs(self.n_channels))
            in_dim = 1 + self.K * self.n_channels + self.K * npairs   # [routing, MF_kc, contrast]
            self.e2e_head = nn.Linear(in_dim, 1).to(dev)

        if self.ablation_mode == "learned_readout":
            # Preprint's QK-only free head (v2.0.0 release, erpxttn.py:318 =
            # self.head = nn.Linear(N*K, 1)): a learned linear over the flattened
            # mean-over-heads cross-attention tensor a ∈ (B, P, K). NO value
            # projection and NO grounded match m enter the decision — this is
            # exactly the ungrounded readout the rebuild replaced.
            self.lr_head = nn.Linear(self.max_peaks * self.K, 1).to(dev)

    # ── shared: everything up to the attention a[b,p,k] (mirrors ERPXTTN.forward) ──
    def _attention(self, x: torch.Tensor):
        B = x.shape[0]
        H, d_h = self.num_heads, self.d_head
        emb, m, mask, center, bounds = self._tokenize_and_match(x)
        P = emb.shape[1]
        z = self.patch_embed(emb) + self._interp_pos(center)
        z = z * mask.unsqueeze(-1)
        z = self.dropout_layer(z)
        if self.use_self_attn:
            z = self._self_attention(z, mask)
        p = self.patch_embed(self.proto_seg) + self.pos_embed[self._proto_patch_idx()]
        p_exp = p.unsqueeze(0).expand(B, -1, -1)
        q = self.W_q(self.ln_q(z)).view(B, P, H, d_h).permute(0, 2, 1, 3)
        k = self.W_k(self.ln_kv(p_exp)).view(B, self.K, H, d_h).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)
        scores = scores.clamp(-10, 10)
        a = F.softmax(scores, dim=-1).mean(dim=1)          # (B, P, K)
        return a, m, mask, p_exp, center, bounds

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        a, m, mask, p_exp, center, bounds = self._attention(x)
        aux = {"a": a, "m": m, "mask": mask, "center": center, "bounds": bounds}

        if self.ablation_mode == "learned_readout":
            # Preprint QK-only free head: learned linear over the flattened
            # attention tensor a (mean over heads), no value projection, no m.
            logit = self.lr_head(a.reshape(a.shape[0], -1))     # (B, P*K) -> (B, 1)
            return logit, aux

        # Grounded routing contribution (shared by nowhiten + e2e). For nowhiten,
        # m is the raw-cosine match because the whitener is identity.
        contrib = (a * m) * mask.unsqueeze(-1)
        n_valid = mask.sum(dim=1).clamp_min(1).float()
        routing_logit = self.match_scale * contrib.sum(dim=(1, 2)) / n_valid   # (B,)

        if self.ablation_mode == "e2e":
            # Joint learnable head over [routing_logit, MF_kc, contrast]; MF factors
            # are fixed projections (no grad), the head + attention train together.
            mf_c, ctr = self.compute_mf_channel(x)              # (B,K,C), (B,K,pairs)
            feats = torch.cat([routing_logit.unsqueeze(1),
                               mf_c.reshape(B, -1), ctr.reshape(B, -1)], dim=1)
            logit = self.e2e_head(feats)                        # (B, 1)
            return logit, aux

        # nowhiten
        return routing_logit.unsqueeze(-1) + self.match_bias, aux


def load_frozen_ablation(ckpt: dict, device):
    """Reload a frozen ERPXTTNAblation from a checkpoint (for the cert contrast on
    learned_readout / e2e, which the base load_frozen_model can't reconstruct). Not
    used by the standard fusion path — nowhiten reloads fine via erpxttn.load_frozen_model."""
    c = ckpt["model_config"]
    sd = ckpt["state_dict"]
    K = int(sd["proto_seg"].shape[0])
    model = ERPXTTNAblation(
        c["n_channels"], c["n_times"], channel_names=c.get("channel_names"),
        detection_channel=c.get("detection_channel"),
        d_model=c.get("d_model", 64), num_heads=c.get("num_heads", 4),
        patch_width=c.get("patch_width", 8), max_k=K, max_peaks=c["max_peaks"],
        use_self_attn=c["use_self_attn"], sfreq=c.get("sfreq", 256.0),
        peak_prominence=c.get("peak_prominence", 0.02),
        ablation_mode=c.get("ablation_mode", "nowhiten"))
    if c.get("ablation_mode") == "e2e":
        npairs = len(channel_pairs(model.n_channels))
        model.e2e_head = nn.Linear(1 + K * model.n_channels + K * npairs, 1)
    if c.get("ablation_mode") == "learned_readout":
        model.lr_head = nn.Linear(model.max_peaks * K, 1)
    model.load_state_dict(sd)
    return model.eval().to(device)
