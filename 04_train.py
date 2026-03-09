#!/usr/bin/env python
"""04_train.py — LOSO cross-validation training for EEGNet and ERP-XTTN.

Usage:
    python 04_train.py --dataset bnci --channels midline3 --model eegnet
    python 04_train.py --dataset hri  --channels midline3 --model erpxttn

Training logs go to: logs/<timestamp>_<dataset>_<channels>_<model>/
Results go to:       datasets/<name>/results/tmin0ms_tmax800ms/<variant>/<model>/
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from eegnet import EEGNet
from erpxttn import ERPXTTN

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 250
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
LR_SCHEDULE_TOTAL = 100
GRAD_CLIP = 1.0
JITTER_MAX = 10
NOISE_SCALE = 0.1
VAL_FRACTION = 0.15

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"

DATASET_CONFIG = {
    "bnci": {
        "name": "bnci_horizon_2020_ErrP",
        "subjects": ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"],
        "variants": {
            "midline2": "noref_midline2_rs256_iir_fwd_bp-1-10",
            "midline3": "noref_midline3_rs256_iir_fwd_bp-1-10",
            "full":     "noref_rs256_iir_fwd_bp-1-10",
        },
    },
    "hri": {
        "name": "hri_cursor",
        "subjects": [f"sub-{i:02d}" for i in [2,3,4,5,6,7,8,9,10,11,13]],
        "variants": {
            "midline2": "noref_midline2_iir_fwd_bp-1-10",
            "midline3": "noref_midline3_iir_fwd_bp-1-10",
            "full":     "noref_iir_fwd_bp-1-10",
        },
    },
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_all_subjects(dataset_key: str, channel_config: str
                      ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """Load all subjects' data from epoched FIF files.

    Returns:
        ({subject_id: (X, y)}, srate)
        where X is (n_epochs, C, T), y is (n_epochs,), srate is sampling rate in Hz
    """
    cfg = DATASET_CONFIG[dataset_key]
    variant = cfg["variants"][channel_config]
    base = DATASETS_DIR / cfg["name"] / "epoched_fif" / "tmin0ms_tmax800ms" / variant

    data = {}
    srate = None
    for subj in cfg["subjects"]:
        all_X, all_y = [], []
        subj_dir = base / subj
        for fif_path in sorted(subj_dir.rglob("*-epo.fif")):
            epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
            if srate is None:
                srate = int(epochs.info["sfreq"])
            event_id = epochs.event_id
            error_ids = {v for k, v in event_id.items() if "error" in k.lower()}
            correct_ids = {v for k, v in event_id.items() if "correct" in k.lower()}
            keep_ids = error_ids | correct_ids
            if not keep_ids:
                continue

            X = epochs.get_data()
            event_codes = epochs.events[:, 2]
            mask = np.isin(event_codes, list(keep_ids))
            X = X[mask]
            codes = event_codes[mask]
            y = np.array([1 if c in error_ids else 0 for c in codes])
            all_X.append(X)
            all_y.append(y)

        X_subj = np.concatenate(all_X, axis=0).astype(np.float32)
        y_subj = np.concatenate(all_y, axis=0).astype(np.int64)
        data[subj] = (X_subj, y_subj)
        logging.info(f"  {subj}: {len(y_subj)} epochs "
                     f"({y_subj.sum()} error, {(1-y_subj).sum()} correct)")
    return data, srate


# ──────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────

def compute_channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std from (N, C, T) array."""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    std = np.clip(std, 1e-8, None)
    return mean, std


# ──────────────────────────────────────────────────────────────────────
# Augmentation (applied on-the-fly)
# ──────────────────────────────────────────────────────────────────────

class AugmentedDataset(torch.utils.data.Dataset):
    """Wraps tensors with on-the-fly temporal jitter + Gaussian noise."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        C, T = x.shape

        shift = torch.randint(-JITTER_MAX, JITTER_MAX + 1, (1,)).item()
        if shift != 0:
            x_new = torch.zeros_like(x)
            if shift > 0:
                x_new[:, shift:] = x[:, :T - shift]
            else:
                x_new[:, :T + shift] = x[:, -shift:]
            x = x_new

        x = x + NOISE_SCALE * torch.randn_like(x)
        return x, self.y[idx]


# ──────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────

def get_lr(epoch: int, total_epochs: int) -> float:
    """Linear warmup + cosine annealing."""
    lr_start = LR / 10
    lr_end = LR / 100

    if epoch < WARMUP_EPOCHS:
        return lr_start + (epoch / WARMUP_EPOCHS) * (LR - lr_start)
    else:
        progress = (epoch - WARMUP_EPOCHS) / (total_epochs - WARMUP_EPOCHS)
        progress = min(progress, 1.0)
        return lr_end + 0.5 * (LR - lr_end) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ──────────────────────────────────────────────────────────────────────
# Training loops
# ──────────────────────────────────────────────────────────────────────

def make_model(model_name: str, n_channels: int, n_times: int,
               srate: int, device: torch.device,
               channel_names: list[str] = None):
    if model_name == "eegnet":
        return EEGNet(n_channels, n_times, srate=srate).to(device)
    elif model_name == "erpxttn":
        return ERPXTTN(
            n_channels, n_times, channel_names=channel_names,
            sfreq=float(srate),
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, optimizer, criterion, device, model_name):
    model.train()
    total_loss, n = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
        optimizer.zero_grad()

        if model_name == "erpxttn":
            logits, _ = model(X_batch)
        else:
            logits = model(X_batch)
        logits = logits.squeeze(-1)

        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if model_name == "eegnet":
            model.apply_weight_constraint()

        total_loss += loss.item() * len(y_batch)
        n += len(y_batch)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, model_name):
    """Evaluate model and return metrics + raw outputs for visualization.

    Returns:
        auroc, bal_acc, probs (N,), labels (N,), attn (N, H, N_patches, K) or None
    """
    model.eval()
    all_logits, all_labels = [], []
    all_attn = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        if model_name == "erpxttn":
            logits, attn = model(X_batch)
            all_attn.append(attn.cpu())
        else:
            logits = model(X_batch)
        all_logits.append(logits.squeeze(-1).cpu())
        all_labels.append(y_batch)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    auroc = roc_auc_score(labels, probs)
    bal_acc = balanced_accuracy_score(labels, preds)

    attn_out = torch.cat(all_attn).numpy() if all_attn else None
    return auroc, bal_acc, probs, labels, attn_out


def run_fold(fold_idx: int, test_subj: str,
             all_data: dict, model_name: str, srate: int,
             device: torch.device, results_dir: Path,
             channel_names: list[str] = None) -> dict:
    """Run one LOSO fold: Phase 1 (find best epoch) + Phase 2 (retrain)."""

    logging.info(f"\n{'='*60}")
    logging.info(f"Fold {fold_idx}: test={test_subj}")
    logging.info(f"{'='*60}")

    # --- Collect train/test data ---
    train_subjects = [s for s in all_data if s != test_subj]
    X_test, y_test = all_data[test_subj]
    X_train_pool = np.concatenate([all_data[s][0] for s in train_subjects])
    y_train_pool = np.concatenate([all_data[s][1] for s in train_subjects])
    subj_labels = np.concatenate([
        np.full(len(all_data[s][1]), i)
        for i, s in enumerate(train_subjects)
    ])

    n_channels, n_times = X_train_pool.shape[1], X_train_pool.shape[2]

    logging.info(f"  Train pool: {len(y_train_pool)} epochs "
                 f"({y_train_pool.sum()} error, {(1-y_train_pool).sum()} correct)")
    logging.info(f"  Test:       {len(y_test)} epochs "
                 f"({y_test.sum()} error, {(1-y_test).sum()} correct)")

    # ── Phase 1: find best_epoch with val split ──

    strat_key = subj_labels * 2 + y_train_pool
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION,
                                      random_state=SEED)
    train_idx, val_idx = next(splitter.split(X_train_pool, strat_key))

    X_train = X_train_pool[train_idx]
    y_train = y_train_pool[train_idx]
    X_val = X_train_pool[val_idx]
    y_val = y_train_pool[val_idx]

    mean, std = compute_channel_stats(X_train)
    X_train_n = ((X_train - mean) / std).astype(np.float32)
    X_val_n = ((X_val - mean) / std).astype(np.float32)

    n_correct = (y_train == 0).sum()
    n_error = (y_train == 1).sum()
    pos_weight = torch.tensor([n_correct / n_error], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = AugmentedDataset(torch.from_numpy(X_train_n),
                                 torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_n),
                            torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    set_seed(SEED)
    model = make_model(model_name, n_channels, n_times, srate, device,
                       channel_names=channel_names)

    if model_name == "erpxttn":
        model.set_prototypes(torch.from_numpy(X_train_n), torch.from_numpy(y_train))
        logging.info(f"  Phase 1 detected windows (ms): {model.detected_windows_ms}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_auroc, best_epoch = -1.0, 0
    epochs_no_improve = 0
    phase1_curves = {"loss": [], "val_auroc": [], "val_bal_acc": []}

    logging.info("  Phase 1: finding best epoch...")
    for epoch in range(MAX_EPOCHS):
        set_lr(optimizer, get_lr(epoch, LR_SCHEDULE_TOTAL))
        loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                device, model_name)
        auroc, bal_acc, _, _, _ = evaluate(model, val_loader, device, model_name)

        phase1_curves["loss"].append(float(loss))
        phase1_curves["val_auroc"].append(float(auroc))
        phase1_curves["val_bal_acc"].append(float(bal_acc))

        if auroc > best_auroc:
            best_auroc = auroc
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epochs_no_improve == 0:
            logging.info(f"    Epoch {epoch:3d}  loss={loss:.4f}  "
                         f"val_auroc={auroc:.4f}  val_bal_acc={bal_acc:.4f}  "
                         f"best={best_auroc:.4f}@{best_epoch}")

        if epochs_no_improve >= PATIENCE:
            logging.info(f"    Early stop at epoch {epoch} "
                         f"(best={best_auroc:.4f}@{best_epoch})")
            break

    retrain_epochs = best_epoch + 1
    logging.info(f"  Phase 1 done: best_epoch={best_epoch}, "
                 f"best_val_auroc={best_auroc:.4f}, retrain for {retrain_epochs} epochs")

    # ── Phase 2: retrain on full pool ──

    mean2, std2 = compute_channel_stats(X_train_pool)
    X_pool_n = ((X_train_pool - mean2) / std2).astype(np.float32)
    X_test_n = ((X_test - mean2) / std2).astype(np.float32)

    n_correct2 = (y_train_pool == 0).sum()
    n_error2 = (y_train_pool == 1).sum()
    pos_weight2 = torch.tensor([n_correct2 / n_error2], dtype=torch.float32).to(device)
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)

    pool_ds = AugmentedDataset(torch.from_numpy(X_pool_n),
                                torch.from_numpy(y_train_pool))
    pool_loader = DataLoader(pool_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_ds = TensorDataset(torch.from_numpy(X_test_n),
                             torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    set_seed(SEED)
    model2 = make_model(model_name, n_channels, n_times, srate, device,
                        channel_names=channel_names)

    if model_name == "erpxttn":
        model2.set_prototypes(torch.from_numpy(X_pool_n),
                               torch.from_numpy(y_train_pool))
        logging.info(f"  Phase 2 detected windows (ms): {model2.detected_windows_ms}")

    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    phase2_curves = {"loss": []}

    logging.info(f"  Phase 2: retraining for {retrain_epochs} epochs on full pool...")
    for epoch in range(retrain_epochs):
        set_lr(optimizer2, get_lr(epoch, LR_SCHEDULE_TOTAL))
        loss = train_one_epoch(model2, pool_loader, optimizer2, criterion2,
                                device, model_name)
        phase2_curves["loss"].append(float(loss))
        if epoch % 10 == 0 or epoch == retrain_epochs - 1:
            logging.info(f"    Epoch {epoch:3d}  loss={loss:.4f}")

    # ── Evaluate on test subject ──
    auroc, bal_acc, probs, labels, attn = evaluate(model2, test_loader, device, model_name)
    logging.info(f"  Test: AUROC={auroc:.4f}  BalAcc={bal_acc:.4f}")

    # Save fold results
    fold_result = {
        "fold": fold_idx,
        "test_subject": test_subj,
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_auroc),
        "test_auroc": float(auroc),
        "test_bal_acc": float(bal_acc),
    }
    if model_name == "erpxttn":
        fold_result["detected_windows_ms"] = model2.detected_windows_ms

    preds_path = results_dir / f"predictions_{test_subj}.npz"
    np.savez_compressed(str(preds_path),
                        probs=probs,
                        labels=labels,
                        auroc=auroc,
                        bal_acc=bal_acc)
    logging.info(f"  Saved predictions to {preds_path}")

    curves_path = results_dir / f"curves_{test_subj}.npz"
    np.savez_compressed(str(curves_path),
                        phase1_loss=phase1_curves["loss"],
                        phase1_val_auroc=phase1_curves["val_auroc"],
                        phase1_val_bal_acc=phase1_curves["val_bal_acc"],
                        phase2_loss=phase2_curves["loss"],
                        best_epoch=best_epoch)
    logging.info(f"  Saved training curves to {curves_path}")

    if attn is not None:
        attn_path = results_dir / f"attention_{test_subj}.npz"
        np.savez_compressed(str(attn_path),
                            attention_weights=attn,
                            labels=labels,
                            auroc=auroc)
        logging.info(f"  Saved attention weights to {attn_path}")

        proto_path = results_dir / f"prototypes_{test_subj}.npz"
        np.savez_compressed(str(proto_path),
                            proto_raw=model2.proto_raw.cpu().numpy(),
                            proto_windows_ms=np.array(model2.detected_windows_ms),
                            sfreq=srate)
        logging.info(f"  Saved prototypes to {proto_path}")

    return fold_result


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LOSO training for ErrP classifiers")
    parser.add_argument("--dataset", required=True, choices=["bnci", "hri"])
    parser.add_argument("--channels", required=True, choices=["midline2", "midline3", "full"])
    parser.add_argument("--model", required=True, choices=["eegnet", "erpxttn"])
    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.dataset}_{args.channels}_{args.model}"
    log_dir = REPO_ROOT / "logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    cfg = DATASET_CONFIG[args.dataset]
    variant = cfg["variants"][args.channels]
    results_dir = (DATASETS_DIR / cfg["name"] / "results" / "tmin0ms_tmax800ms"
                   / variant / args.model)
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Run: {run_name}")
    logging.info(f"Args: {vars(args)}")
    logging.info(f"Log dir: {log_dir}")
    logging.info(f"Results dir: {results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    set_seed(SEED)

    logging.info(f"Loading {args.dataset} / {args.channels}...")
    all_data, srate = load_all_subjects(args.dataset, args.channels)
    logging.info(f"Sampling rate: {srate} Hz")
    subjects = list(all_data.keys())

    # Read channel names (needed for ERP-XTTN Cz detection)
    channel_names = None
    if args.model == "erpxttn":
        base = (DATASETS_DIR / cfg["name"] / "epoched_fif" / "tmin0ms_tmax800ms"
                / variant)
        first_fif = next((base / cfg["subjects"][0]).rglob("*-epo.fif"))
        channel_names = mne.read_epochs(str(first_fif), preload=False,
                                         verbose=False).ch_names
        logging.info(f"Channels ({len(channel_names)}): {channel_names}")

    # LOSO cross-validation
    results = []
    t0 = time.time()
    for i, test_subj in enumerate(subjects):
        fold_result = run_fold(i, test_subj, all_data, args.model, srate,
                               device, results_dir,
                               channel_names=channel_names)
        results.append(fold_result)

    elapsed = time.time() - t0

    aurocs = [r["test_auroc"] for r in results]
    bal_accs = [r["test_bal_acc"] for r in results]

    logging.info(f"\n{'='*60}")
    logging.info(f"RESULTS: {args.dataset} / {args.channels} / {args.model}")
    logging.info(f"{'='*60}")
    for r in results:
        extra = ""
        if "detected_windows_ms" in r:
            extra = f"  windows={r['detected_windows_ms']}"
        logging.info(f"  {r['test_subject']}: AUROC={r['test_auroc']:.4f}  "
                     f"BalAcc={r['test_bal_acc']:.4f}  "
                     f"best_epoch={r['best_epoch']}{extra}")
    logging.info(f"  Mean AUROC:   {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    logging.info(f"  Mean BalAcc:  {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
    logging.info(f"  Total time:   {elapsed:.0f}s ({elapsed/60:.1f}min)")

    summary = {
        "run_name": run_name,
        "args": vars(args),
        "seed": SEED,
        "device": str(device),
        "elapsed_seconds": round(elapsed, 1),
        "folds": results,
        "mean_auroc": round(float(np.mean(aurocs)), 4),
        "std_auroc": round(float(np.std(aurocs)), 4),
        "mean_bal_acc": round(float(np.mean(bal_accs)), 4),
        "std_bal_acc": round(float(np.std(bal_accs)), 4),
    }
    for d in [results_dir, log_dir]:
        with open(d / "results.json", "w") as f:
            json.dump(summary, f, indent=2)

    logging.info(f"Results saved to {results_dir / 'results.json'}")
    logging.info(f"Log saved to {log_dir / 'train.log'}")


if __name__ == "__main__":
    main()
