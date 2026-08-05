"""ablation_val_learned.py — grounding test for the learned free-head ablation.

The grounded model's decision is Σ a·m, so corrupting a template drives its AUROC
below chance (the swap ladder collapses). The free head's decision is lr_head(a),
which does NOT use m — so the SAME template corruptions leave its AUROC essentially
unchanged (flat ladder). Running the ablation model's FORWARD under each corruption
demonstrates that directly, and is the "grounded vs ungrounded readout" contrast.

Only runs on datasets where learned_readout was actually trained (the ablation set).
Writes validation_learned.json next to the learned_readout results for the dashboard.

Usage:  python ablation_val_learned.py --dataset hri_errp_cursor --seed 1
        (loops the 4 ablation datasets x 5 seeds if no args, skipping absent ones)
"""
import argparse, importlib.util, glob, json, os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ablation_erpxttn import load_frozen_ablation

REPO = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("val08", REPO / "08_validate.py")
val08 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(val08)
_tr = importlib.util.spec_from_file_location("t4", REPO / "04_train.py")
train = importlib.util.module_from_spec(_tr); _tr.loader.exec_module(train)

DATASETS = {"hri_errp_cursor": "midline3", "erpcore_ern": "midline3_ern",
            "erpcore_p300": "midline3", "erpcore_n400": "midline3_n400"}


@torch.no_grad()
def _fwd_auroc(model, Xn, y, device):
    lg = model(torch.from_numpy(Xn).float().to(device))[0].squeeze(-1).cpu().numpy()
    return val08._auroc(y, lg)


def run(dataset, seed, device):
    key = DATASETS[dataset]
    cfg = train.load_dataset_config(dataset)
    variant = cfg["variants"][key]
    rd = REPO / "datasets" / cfg["name"] / "results" / "tmin0ms_tmax800ms" / variant / "ablation_erpxttn_learned_readout" / f"seed-{seed}"
    if not rd.exists() or not list(rd.glob("checkpoint_*.pt")):
        print(f"  {dataset} seed-{seed}: no learned_readout checkpoints yet, skip"); return None
    all_data, _ = train.load_all_subjects(cfg, key)
    rng = np.random.default_rng(val08.RNG_SEED)
    rows = {"intact": [], "noise": [], "polarity": [], "cross_component": []}
    for s in all_data:
        ck = rd / f"checkpoint_{s}.pt"
        if not ck.exists():
            continue
        c = torch.load(str(ck), map_location=device, weights_only=False)
        model = load_frozen_ablation(c, device)
        X, y = all_data[s]
        Xn = ((X - c["norm_mean"]) / c["norm_std"]).astype(np.float32)
        if len(np.unique(y)) < 2:
            continue
        rows["intact"].append(_fwd_auroc(model, Xn, y, device))
        rows["noise"].append(_fwd_auroc(val08.swap_same_window(model, "noise", rng), Xn, y, device))
        rows["polarity"].append(_fwd_auroc(val08.swap_same_window(model, "polarity", rng), Xn, y, device))
        rows["cross_component"].append(_fwd_auroc(val08.swap_cross_component(model), Xn, y, device))
    if not rows["intact"]:
        return None
    agg = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
               "n_folds": len(v)} for k, v in rows.items()}
    out = {"model": "learned_readout", "readout": "free head lr_head(a) — decision ignores m",
           "n_folds": len(rows["intact"]), "ladder_forward": agg,
           "note": "ladder is flat because the free head does not read the match m; "
                   "grounded model (validation.json) collapses below chance."}
    (rd / "validation_learned.json").write_text(json.dumps(out, indent=2))
    l = agg
    print(f"  {dataset} seed-{seed}: forward ladder intact={l['intact']['mean']:.3f} "
          f"polarity={l['polarity']['mean']:.3f} cross={l['cross_component']['mean']:.3f} "
          f"(flat = ungrounded)")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_list = [a.dataset] if a.dataset else list(DATASETS)
    seeds = [a.seed] if a.seed else [1, 2, 3, 4, 5]
    for ds in ds_list:
        for s in seeds:
            run(ds, s, device)


if __name__ == "__main__":
    main()
