"""ablation_amp_only.py — the `amp_only` arm of the ablation grid.

Drops the routing logit from the two-factor fusion vector: φ = [MF_kc, contrast]
(channel-resolved + bipolar amplitude only, no routing). Because MF_kc / contrast
are fixed projections of the frozen prototypes — independent of the learned
attention — this needs NO retraining: it just refits the per-fold zero-calibration
LOSO combiner on the base `erpxttn_peak` checkpoints, using the amplitude columns
only. Writes to a TAGGED dir (`erpxttn_peak_amp_only/`) so the base 3ch is untouched.

Usage:  python ablation_amp_only.py --dataset hri_errp_cursor --seed 1
        (loops all 4 ablation datasets × 5 seeds if no args)
"""
import argparse, json, os, importlib.util
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from erpxttn import load_frozen_model, fusion_feature_metadata

REPO = Path(__file__).resolve().parent
ABLATION_DATASETS = ["erpcore_ern", "hri_errp_cursor", "erpcore_p300", "erpcore_n400"]


def _load_train():
    spec = importlib.util.spec_from_file_location("t4", REPO / "04_train.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@torch.no_grad()
def _amp_features(model, Xn, device, bs=256):
    """Amplitude-only fusion features per trial: [MF_kc, contrast] (NO routing logit)."""
    feats = []
    for i in range(0, len(Xn), bs):
        xb = torch.from_numpy(Xn[i:i + bs]).to(device)
        B = xb.shape[0]
        kc, ct = model.compute_mf_channel(xb)
        feats.append(np.column_stack([kc.reshape(B, -1).cpu().numpy(),
                                      ct.reshape(B, -1).cpu().numpy()]))
    return np.concatenate(feats)


def run(dataset, seed, device):
    train = _load_train()
    cfg = train.load_dataset_config(dataset)
    ch = [k for k in cfg["variants"] if k != "full"][0]
    all_data, _ = train.load_all_subjects(cfg, ch)
    variant = cfg["variants"][ch]
    base = REPO / "datasets" / cfg["name"] / "results" / "tmin0ms_tmax800ms" / variant / "erpxttn_peak" / f"seed-{seed}"
    out = REPO / "datasets" / cfg["name"] / "results" / "tmin0ms_tmax800ms" / variant / "erpxttn_peak_amp_only" / f"seed-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    subs = list(all_data)
    if not all((base / f"checkpoint_{s}.pt").exists() for s in subs):
        print(f"  {dataset} seed-{seed}: base checkpoints missing, skip"); return None

    per = {}
    for s in subs:
        ckpt = torch.load(str(base / f"checkpoint_{s}.pt"), map_location=device, weights_only=False)
        f_s = load_frozen_model(ckpt, device)
        nm, ns = ckpt["norm_mean"], ckpt["norm_std"]
        Xtr, ytr = [], []
        for o in subs:
            if o == s:
                continue
            Xo, yo = all_data[o]
            Xtr.append(_amp_features(f_s, ((Xo - nm) / ns).astype(np.float32), device)); ytr.append(yo)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(np.concatenate(Xtr), np.concatenate(ytr))
        Xs, ys = all_data[s]
        prob = clf.predict_proba(_amp_features(f_s, ((Xs - nm) / ns).astype(np.float32), device))[:, 1]
        per[s] = float(roc_auc_score(ys, prob))
        np.savez_compressed(str(out / f"two_factor_{s}.npz"), probs=prob, labels=ys, auroc=per[s],
                            coef=clf.coef_[0], intercept=clf.intercept_,
                            combiner_features="amp_only_channel_contrast")

    # results.json mirroring the base schema so the dashboard reads it uniformly.
    base_json = json.load(open(base / "results.json"))
    vals = [per[s] for s in subs]
    summary = dict(base_json)  # inherit fold structure / routing numbers
    summary["ablation"] = "amp_only"
    summary["mean_two_factor_auroc"] = round(float(np.mean(vals)), 4)
    summary["std_two_factor_auroc"] = round(float(np.std(vals)), 4)
    summary["two_factor_auroc_per_subject"] = {s: round(per[s], 4) for s in subs}
    summary["two_factor_combiner"] = "amp_only_channel_contrast_v2"
    json.dump(summary, open(out / "results.json", "w"), indent=2)
    print(f"  {dataset} seed-{seed}: amp_only mean AUROC = {np.mean(vals):.4f}  (routing-only base: "
          f"{base_json.get('mean_routing_auroc'):.4f}, full fusion base: {base_json.get('mean_two_factor_auroc'):.4f})")
    return np.mean(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_list = [a.dataset] if a.dataset else ABLATION_DATASETS
    seeds = [a.seed] if a.seed else [1, 2, 3, 4, 5]
    for ds in ds_list:
        for s in seeds:
            run(ds, s, device)


if __name__ == "__main__":
    main()
