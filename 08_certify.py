#!/usr/bin/env python
"""08_certify.py — faithfulness certificate battery for the two-factor ERP-XTTN.

Loads the FROZEN per-fold routing checkpoints emitted by 04_train.py and runs the
§1 faithfulness interventions on them at INFERENCE TIME — no retraining. The
grounded ``logit = match_scale · Σ a·m / #peaks`` readout cannot compensate for a
corrupted template the way a free head could, so an inference-time template swap
is a valid necessity test (that is the whole payoff of the grounded design; no
retrain-from-random control is needed).

Battery (checklist §1):
  routing (§1a):
    - G_a / G_m / G_c grounding: contribution concentration of c = a·m, with an
      empirical null (shuffle the peak↔proto association) + CI. Expect G_c
      localizes; G_m alone ≈ null (the match doesn't localize by itself).
    - causal occlusion M-sweep: occlude the top-|c| (peak,proto) cells vs random.
    - frozen-swap ladder: gaussian-noise → phase-randomized → time-reversed →
      polarity-flip → cross-component, vs a null-template reference. Same-window
      rungs keep that window's whitening; cross-component uses the TARGET
      window's whitening. Expect monotonic degradation + below-chance inversion
      on polarity/cross.
    - carrier scrambles: break peak↔proto and trial↔trial correspondence → chance.
    - proto-drop: per-component ΔAUROC (expect redundancy).
  amplitude (§1b): window-localization (on / off / early-baseline) and
    permute-trial on the matched-filter factor.
  combined (§1c): routing-only vs two-factor AUROC, and top-M grounded
    sufficiency (R²) on the routing logit.

Fold-level variance is reported per check; run once per seed and combine across
seeds for the seed-level variance (a checkpoint exists per seed × dataset).

Usage:
    python 08_certify.py --dataset erpcore_p300 --channels midline3 --seed 1
    python 08_certify.py --self-test          # synthetic model+data, no fif needed
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from erpxttn import ERPXTTN, load_frozen_model, _t_resample, _fold_features

REPO = Path(__file__).resolve().parent
RNG_SEED = 20260722


# ──────────────────────────────────────────────────────────────────────
# Model reconstruction + grounded forward
# ──────────────────────────────────────────────────────────────────────

# load_frozen_model is defined in erpxttn.py (shared with the Stage-2 fusion) and
# imported above; the certificate reloads each fold's frozen checkpoint with it.


@torch.no_grad()
def grounded_forward(model: ERPXTTN, X: np.ndarray, device):
    """Return (logits (N,), a (N,P,K), m (N,P,K), mask (N,P)) for test epochs X."""
    logit, aux = model(torch.from_numpy(X).float().to(device))
    return (logit.squeeze(-1).cpu().numpy(), aux["a"].cpu().numpy(),
            aux["m"].cpu().numpy(), aux["mask"].cpu().numpy())


def readout(scale: float, a, m, mask):
    """Grounded readout from (possibly intervened) a, m, mask — matches the model."""
    contrib = a * m * mask[..., None]
    n_valid = np.clip(mask.sum(axis=1), 1, None)
    return scale * contrib.sum(axis=(1, 2)) / n_valid


@torch.no_grad()
def _forward_full(model: ERPXTTN, X: np.ndarray, device):
    """grounded_forward plus the per-peak window bounds (for template swaps)."""
    logit, aux = model(torch.from_numpy(X).float().to(device))
    return (logit.squeeze(-1).cpu().numpy(), aux["a"].cpu().numpy(),
            aux["m"].cpu().numpy(), aux["mask"].cpu().numpy(),
            aux["bounds"].cpu().numpy())


def _match_from_bounds(model, X, bounds, mask, device):
    """Recompute the whitened-cosine match m from FIXED peak bounds and a (swapped)
    model's match buffers — no re-detection. Uses the model's own on-device torch
    path (`_t_resample`, fp32), so it is numerically identical to the model's own
    match; the ladder can tokenize once and only recompute m per swap.

    X/bounds/mask are the numpy intermediates from _forward_full; m is returned as
    numpy for the numpy `readout`.
    """
    Xt = torch.from_numpy(X).to(device)
    bt = torch.from_numpy(bounds).to(device)
    mt = torch.from_numpy(mask).to(device)
    B, P = mask.shape
    C, K = model.n_channels, int(model.proto_w.shape[0])
    m = torch.zeros(B, P, K, device=device)
    bidx, jidx = torch.where(mt)
    if bidx.numel() > 0:
        los, his = bt[bidx, jidx, 0], bt[bidx, jidx, 1]
        for k in range(K):
            wk = int(model.proto_w[k].item())
            if wk <= 0:
                continue
            dk = C * wk
            seg = _t_resample(Xt, bidx, los, his, wk).reshape(-1, dk)
            V = seg @ model.whitener[k, :dk, :dk].T
            vn = V.norm(dim=1) + 1e-8
            m[bidx, jidx, k] = (V @ model.proto_white[k, :dk]) / (vn * model.proto_white_norm[k])
    return (m * mt.unsqueeze(-1)).cpu().numpy()


def _auroc(y, scores):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


# ──────────────────────────────────────────────────────────────────────
# Template interventions (build a frozen copy with swapped templates)
# ──────────────────────────────────────────────────────────────────────

def _phase_randomize(tmpl: np.ndarray, rng) -> np.ndarray:
    """Randomize phase per channel, preserving magnitude spectrum."""
    F = np.fft.rfft(tmpl, axis=-1)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape))
    ph[..., 0] = 1.0
    out = np.fft.irfft(np.abs(F) * ph, n=tmpl.shape[-1], axis=-1)
    return out.astype(np.float32)


def _rebuild_slot(model, m2, k, new_tmpl, s, e, whitener_block):
    """Write a transformed native template into the MATCH buffers of slot k.

    The routing key (proto_seg) and PE (proto_center) are left INTACT so the
    frozen routing `a` is unchanged; only the grounded match m is corrupted.
    This is the necessity test: a fixed `a` cannot compensate for a corrupted m.
    """
    C = model.n_channels
    d = C * (e - s)
    pw_vec = whitener_block @ new_tmpl.reshape(-1)
    m2.proto_white[k, :d] = torch.from_numpy(pw_vec.astype(np.float32))
    m2.proto_white[k, d:] = 0.0
    m2.proto_white_norm[k] = float(np.linalg.norm(pw_vec) + 1e-8)
    m2.mf_template[k] = 0.0
    m2.mf_template[k, :, s:e] = torch.from_numpy(new_tmpl.astype(np.float32))


def swap_same_window(model, kind, rng):
    """Ladder rungs that keep each prototype's window (and its whitening)."""
    m2 = copy.deepcopy(model)
    for k in range(model.K):
        s, e = int(model.mf_window[k, 0]), int(model.mf_window[k, 1])
        if e <= s:
            continue
        tmpl = model.mf_template[k, :, s:e].cpu().numpy()
        if kind == "noise":
            new = (rng.standard_normal(tmpl.shape) * (tmpl.std() + 1e-6)).astype(np.float32)
        elif kind == "reversed":
            new = tmpl[:, ::-1].copy()
        elif kind == "polarity":
            new = -tmpl
        elif kind == "phase":
            new = _phase_randomize(tmpl, rng)
        else:
            raise ValueError(kind)
        d = model.n_channels * (e - s)
        Wk = model.whitener[k, :d, :d].cpu().numpy()
        _rebuild_slot(model, m2, k, new, s, e, Wk)
    return m2


def swap_cross_component(model):
    """Cross-component rung: peaks routed to slot k (by the intact key) are matched
    against component (k+1)'s template, using the TARGET window's whitening
    (required so the collapse reflects the content swap, not a whitening
    mismatch). Only the MATCH buffers are shifted — the routing key proto_seg is
    left intact, so this is a genuine routing↔match mismatch, not a relabeling of
    the K prototypes (which would leave the summed readout invariant)."""
    m2 = copy.deepcopy(model)
    K = model.K
    # Shift only the match pathway; keep proto_seg / proto_center intact.
    src = {name: getattr(model, name).clone() for name in
           ("proto_white", "proto_white_norm", "whitener", "proto_w",
            "mf_template", "mf_window")}
    for k in range(K):
        j = (k + 1) % K
        for name, buf in src.items():
            getattr(m2, name)[k] = buf[j]
    return m2


# ──────────────────────────────────────────────────────────────────────
# §1a routing certificate
# ──────────────────────────────────────────────────────────────────────

def _concentration(value, mask):
    """Mean over trials of top-cell |value| mass fraction over valid peaks."""
    out = []
    for b in range(value.shape[0]):
        v = np.abs(value[b][mask[b]]).reshape(-1)
        if v.size == 0 or v.sum() == 0:
            continue
        out.append(v.max() / (v.sum() + 1e-12))
    return float(np.mean(out)) if out else float("nan")


def check_grounding(a, m, mask, rng, n_null=200):
    """G_a / G_m / G_c concentration with an empirical null on c."""
    c = a * m
    G_a, G_m, G_c = (_concentration(a, mask), _concentration(m, mask),
                     _concentration(c, mask))
    null = []
    for _ in range(n_null):
        mp = m.copy()
        for b in range(mp.shape[0]):
            idx = np.where(mask[b])[0]
            if len(idx) > 1:
                mp[b, idx] = mp[b, rng.permutation(idx)]
        null.append(_concentration(a * mp, mask))
    null = np.array(null)
    return {"G_a": G_a, "G_m": G_m, "G_c": G_c,
            "G_c_null_mean": float(null.mean()),
            "G_c_null_ci95": [float(np.percentile(null, 2.5)),
                              float(np.percentile(null, 97.5))],
            "G_c_above_null": bool(G_c > np.percentile(null, 97.5))}


def _rng_choice(rng, pool, M):
    if len(pool) == 0:
        return []
    return rng.choice(pool, size=min(M, len(pool)), replace=False)


def check_occlusion(scale, a, m, mask, y, rng, Ms=(1, 2, 3, 5)):
    """Occlude top-|c| (peak,proto) cells vs random (seeded); AUROC drop per M."""
    c = np.abs(a * m) * mask[..., None]
    base = _auroc(y, readout(scale, a, m, mask))
    out = {"baseline_auroc": base, "sweep": []}
    for M in Ms:
        top_m, rnd_m = m.copy(), m.copy()
        for b in range(m.shape[0]):
            flat = c[b].reshape(-1)
            if (flat > 0).sum() == 0:
                continue
            top = np.argsort(flat)[::-1][:M]
            rnd = _rng_choice(rng, np.where(flat > 0)[0], M)
            for cell in top:
                top_m[b, cell // m.shape[2], cell % m.shape[2]] = 0.0
            for cell in rnd:
                rnd_m[b, cell // m.shape[2], cell % m.shape[2]] = 0.0
        out["sweep"].append({
            "M": M,
            "auroc_top": _auroc(y, readout(scale, a, top_m, mask)),
            "auroc_random": _auroc(y, readout(scale, a, rnd_m, mask))})
    return out


def check_proto_drop(scale, a, m, mask, y):
    """Per-component ΔAUROC when prototype k is dropped from the readout."""
    base = _auroc(y, readout(scale, a, m, mask))
    per_k = []
    for k in range(m.shape[2]):
        m2 = m.copy(); m2[:, :, k] = 0.0
        auc = _auroc(y, readout(scale, a, m2, mask))
        per_k.append({"k": k, "auroc": auc, "delta": base - auc})
    return {"baseline_auroc": base, "per_component": per_k}


def check_carrier(scale, a, m, mask, y, rng):
    """Break peak↔proto and trial↔trial correspondence → expect chance."""
    m_pp = m.copy()
    for b in range(m.shape[0]):
        m_pp[b] = m_pp[b][:, rng.permutation(m.shape[2])]   # scramble proto columns
    m_tr = m[rng.permutation(m.shape[0])]                    # scramble trials vs a
    return {"scramble_peak_proto_auroc": _auroc(y, readout(scale, a, m_pp, mask)),
            "scramble_trial_auroc": _auroc(y, readout(scale, a, m_tr, mask))}


def check_swap_ladder(model, X, y, device, rng):
    """Frozen-swap ladder, in rung order. Because swaps corrupt only the match
    (proto_seg / routing a are untouched), the tokenizer runs ONCE: a and the
    peak bounds are shared, and only m is recomputed per swap."""
    scale = float(model.match_scale.item())
    _, a, m_intact, mask, bounds = _forward_full(model, X, device)

    def rung(sw):
        m = _match_from_bounds(sw, X, bounds, mask, device)
        return _auroc(y, readout(scale, a, m, mask))

    rungs = {"intact": _auroc(y, readout(scale, a, m_intact, mask))}
    for kind in ("noise", "phase", "reversed", "polarity"):
        rungs[kind] = rung(swap_same_window(model, kind, rng))
    rungs["cross_component"] = rung(swap_cross_component(model))
    # gaussian-noise doubles as the null-template reference line.
    rungs["null_reference"] = rungs["noise"]
    return rungs


# ──────────────────────────────────────────────────────────────────────
# §1b amplitude (matched-filter) + §1c combined
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _mf_at(model, X, device, shift=0):
    """compute_mf with every prototype window shifted by `shift` samples."""
    m2 = copy.deepcopy(model)
    T = model.n_times
    for k in range(model.K):
        s, e = int(model.mf_window[k, 0]), int(model.mf_window[k, 1])
        w = e - s
        s2 = int(np.clip(s + shift, 0, T - w)); e2 = s2 + w
        m2.mf_window[k, 0] = s2; m2.mf_window[k, 1] = e2
        tmpl = torch.zeros_like(m2.mf_template[k])
        tmpl[:, s2:e2] = model.mf_template[k, :, s:e]
        m2.mf_template[k] = tmpl
        m2.mf_template_norm[k] = float(model.mf_template[k, :, s:e].norm() + 1e-8)
    return m2.compute_mf(torch.from_numpy(X).float().to(device)).cpu().numpy()


def _mf_auroc(mf, y):
    """Best-single-component AUROC of a matched-filter vector (orientation-free)."""
    best = 0.5
    for k in range(mf.shape[1]):
        best = max(best, max(_auroc(y, mf[:, k]), _auroc(y, -mf[:, k])))
    return best


def check_amplitude(model, X, y, device, rng):
    """Window-localization (on / off / early-baseline) + permute-trial on MF."""
    on = _mf_at(model, X, device, shift=0)
    off = _mf_at(model, X, device, shift=int(round(0.15 * model.sfreq)))  # +150 ms
    base = _mf_at(model, X, device, shift=-int(round(0.30 * model.sfreq)))  # early
    perm = on[rng.permutation(on.shape[0])]
    return {"on_component_auroc": _mf_auroc(on, y),
            "off_component_auroc": _mf_auroc(off, y),
            "early_baseline_auroc": _mf_auroc(base, y),
            "permute_trial_auroc": _mf_auroc(perm, y)}


def check_combined_sufficiency(scale, a, m, mask, mf, y, combiner=None,
                               combiner_feats=None, Ms=(1, 2, 3, 5)):
    """Routing-pathway sufficiency + fusion.

    `topM_routing_R2` is ERASER-style sufficiency on the ROUTING logit (top-M
    grounded a·m terms). The headline two-factor AUROC is produced by
    04_train.combine_two_factor (a cross-subject LOSO logistic combiner) and is
    NOT re-derivable per fold — so here:
      * `two_factor_proxy_auroc` is an unweighted routing_logit + Σ MF proxy
        (labelled a proxy so it is never quoted as the headline);
      * `two_factor_auroc` is the TRUE combined logit, present only when this
        fold's fitted combiner weights (coef, intercept) are passed in.
    """
    routing_logit = readout(scale, a, m, mask)
    n_valid = np.clip(mask.sum(axis=1), 1, None)
    var = routing_logit.var() + 1e-12
    r2 = []
    for M in Ms:
        approx = np.zeros_like(routing_logit)
        for b in range(m.shape[0]):
            flat = (a[b] * m[b] * mask[b][:, None]).reshape(-1)
            keep = np.argsort(np.abs(flat))[::-1][:M]
            approx[b] = scale * flat[keep].sum() / n_valid[b]
        r2.append({"M": M, "R2": float(1 - ((routing_logit - approx) ** 2).mean() / var)})

    out = {"routing_only_auroc": _auroc(y, routing_logit),
           "two_factor_proxy_auroc": _auroc(y, routing_logit + mf.sum(axis=1)),
           "topM_routing_R2": r2}
    if combiner is not None:
        coef, intercept = combiner
        # Use the SAME grounded feature vector the combiner was fit on
        # ([routing_logit, MF, MF_kc, contrasts]); fall back to [routing_logit, MF].
        feats = combiner_feats if combiner_feats is not None else np.column_stack([routing_logit, mf])
        combined = feats @ np.asarray(coef) + float(intercept)
        out["two_factor_auroc"] = _auroc(y, combined)
    return out


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def certify_fold(model, X, y, device, rng, combiner=None):
    """Run the full battery on one frozen fold; return a nested dict."""
    logits, a, m, mask = grounded_forward(model, X, device)
    scale = float(model.match_scale.item())
    mf = model.compute_mf(torch.from_numpy(X).float().to(device)).cpu().numpy()
    # Full grounded fusion feature vector (matches the combiner: routing_logit,
    # MF, channel-resolved MF_kc, and bipolar contrasts).
    feats = _fold_features(model, X.astype(np.float32), device) if combiner is not None else None
    return {
        "routing_auroc": _auroc(y, logits),
        "grounding": check_grounding(a, m, mask, rng),
        "occlusion": check_occlusion(scale, a, m, mask, y, rng),
        "proto_drop": check_proto_drop(scale, a, m, mask, y),
        "carrier": check_carrier(scale, a, m, mask, y, rng),
        "swap_ladder": check_swap_ladder(model, X, y, device, rng),
        "amplitude": check_amplitude(model, X, y, device, rng),
        "combined": check_combined_sufficiency(scale, a, m, mask, mf, y, combiner, feats),
    }


def _agg(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_folds": len(vals)}


def aggregate(fold_results):
    """Fold-level mean±sd for the headline scalars of each check."""
    g = lambda path: [_dig(fr, path) for fr in fold_results]
    return {
        "n_folds": len(fold_results),
        "routing_auroc": _agg(g("routing_auroc")),
        "G_c": _agg(g("grounding.G_c")),
        "G_c_null_mean": _agg(g("grounding.G_c_null_mean")),
        "G_m": _agg(g("grounding.G_m")),
        "ladder_intact": _agg(g("swap_ladder.intact")),
        "ladder_polarity": _agg(g("swap_ladder.polarity")),
        "ladder_cross": _agg(g("swap_ladder.cross_component")),
        "ladder_null": _agg(g("swap_ladder.null_reference")),
        "carrier_peak_proto": _agg(g("carrier.scramble_peak_proto_auroc")),
        "carrier_trial": _agg(g("carrier.scramble_trial_auroc")),
        "amp_on": _agg(g("amplitude.on_component_auroc")),
        "amp_off": _agg(g("amplitude.off_component_auroc")),
        "amp_baseline": _agg(g("amplitude.early_baseline_auroc")),
        "amp_permute": _agg(g("amplitude.permute_trial_auroc")),
        "two_factor_proxy": _agg(g("combined.two_factor_proxy_auroc")),
        "two_factor": _agg(g("combined.two_factor_auroc")),
    }


def _dig(d, path):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train04", REPO / "04_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_real(args, device):
    train = _load_train_module()
    cfg = train.load_dataset_config(args.dataset)
    all_data, srate = train.load_all_subjects(cfg, args.channels)
    variant = cfg["variants"][args.channels]
    rdir = (REPO / "datasets" / cfg["name"] / "results" / "tmin0ms_tmax800ms"
            / variant / "erpxttn_peak" / f"seed-{args.seed}")
    rng = np.random.default_rng(RNG_SEED)
    fold_results = []
    for subj in all_data:
        ckpt_path = rdir / f"checkpoint_{subj}.pt"
        if not ckpt_path.exists():
            print(f"  {subj}: no checkpoint, skipping")
            continue
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = load_frozen_model(ckpt, device)
        X, y = all_data[subj]
        Xn = ((X - ckpt["norm_mean"]) / ckpt["norm_std"]).astype(np.float32)
        # This subject's held-out LOSO combiner weights (if 04_train ran the
        # two-factor combiner) → the TRUE combined logit for §1c.
        tf_path = rdir / f"two_factor_{subj}.npz"
        combiner = None
        if tf_path.exists():
            d = np.load(tf_path)
            combiner = (d["coef"], float(d["intercept"]))
        fr = certify_fold(model, Xn, y, device, rng, combiner)
        fold_results.append(fr)
        print(f"  {subj}: routing_auroc={fr['routing_auroc']:.3f} "
              f"G_c={fr['grounding']['G_c']:.3f} "
              f"ladder polarity={fr['swap_ladder']['polarity']:.3f}")
    return fold_results


def run_self_test(device):
    """Synthetic frozen model + separable data — validates the battery end-to-end."""
    rng = np.random.default_rng(0); torch.manual_seed(0)
    C, T, N = 3, 205, 80
    t = np.linspace(0, 1, T)
    X = (rng.standard_normal((N, C, T)) * 0.5).astype(np.float32)
    y = (np.arange(N) % 2).astype(np.float32)
    X[y == 1, 1] += (np.sin(2 * np.pi * 3 * t) * 1.5).astype(np.float32)
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    model = ERPXTTN(C, T, channel_names=["Fz", "Cz", "Pz"], detection_channel="Cz",
                    max_k=4).to(device)
    model.set_prototypes(Xt, yt)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    for _ in range(15):
        model.train(); opt.zero_grad()
        logit, _ = model(Xt); crit(logit.squeeze(-1), yt).backward(); opt.step()
    model.eval()
    fr = certify_fold(model, X, y, device, np.random.default_rng(RNG_SEED))
    return [fr]


def main():
    ap = argparse.ArgumentParser(description="ERP-XTTN faithfulness certificate")
    ap.add_argument("--dataset")
    ap.add_argument("--channels")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.self_test:
        fold_results = run_self_test(device)
    else:
        if not (args.dataset and args.channels):
            ap.error("--dataset and --channels are required (or use --self-test)")
        fold_results = run_real(args, device)

    if not fold_results:
        print("No folds certified (no checkpoints found).")
        return

    summary = {"n_folds": len(fold_results), "aggregate": aggregate(fold_results),
               "folds": fold_results}
    out = args.out or (REPO / f"certificate_{args.dataset or 'selftest'}_"
                       f"{args.channels or 'x'}_seed{args.seed}.json")
    Path(out).write_text(json.dumps(summary, indent=2))

    agg = summary["aggregate"]
    print("\n=== Certificate (fold-level mean) ===")
    for key in ("routing_auroc", "G_c", "G_c_null_mean", "G_m", "ladder_intact",
                "ladder_polarity", "ladder_cross", "carrier_peak_proto",
                "carrier_trial", "amp_on", "amp_off", "amp_baseline",
                "amp_permute", "two_factor_proxy", "two_factor"):
        v = agg.get(key)
        if v:
            print(f"  {key:22s} {v['mean']:.3f} ± {v['sd']:.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
