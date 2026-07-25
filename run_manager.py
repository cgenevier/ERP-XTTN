#!/usr/bin/env python
"""
Run Manager for ERP-XTTN experiments (multi-seed revision sweep).

Enumerates the full experiment matrix — {EEGNet, EEG-Deformer, EPMN, ERP-XTTN}
across 9 datasets x {3-channel, full} x 5 seeds, plus the deterministic
xDAWN+RG (1 seed), plus the ERP-XTTN ablation grid on HRI/P300/N400 (3-channel)
— checks status, and launches runs up to MAX_GPU_JOBS.

Usage:
    python run_manager.py                    # Status check only (3ch main)
    python run_manager.py --launch           # Launch next batch if slots free
    python run_manager.py --launch-all       # Keep launching until queue full
    python run_manager.py --daemon           # Loop: check every 5min, auto-launch
    python run_manager.py --include-full     # Add full-montage runs
    python run_manager.py --include-ablation  # Add the ablation grid
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
LOGS_DIR = REPO_ROOT / "logs"
TRAIN_SCRIPT = REPO_ROOT / "04_train.py"
PYTHON = sys.executable
MAX_GPU_JOBS = 20
DAEMON_INTERVAL = 300  # seconds between daemon checks

# Seeds for stochastic (neural) models. xDAWN+RG is deterministic -> seed 1 only.
SEEDS = [1, 2, 3, 4, 5]
XDAWN_SEED = 1

# Thread limit for launched processes
THREAD_ENV = {
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
}

# Models that don't need a GPU slot
CPU_ONLY_MODELS = {"xdawn_rg"}
NEURAL_MODELS = ["eegnet", "eeg_deformer", "epmn", "erpxttn"]

# ── Dataset / channel-preset table ─────────────────────────────────────
# (label, dataset_id, 3ch_preset)
DATASETS = [
    ("BNCI", "bnci_errp_013-2015", "midline3"),
    ("HRI",  "hri_errp_cursor",    "midline3"),
    ("N400", "erpcore_n400",       "midline3_n400"),
    ("P300", "erpcore_p300",       "midline3"),
    ("N2pc", "erpcore_n2pc",       "posterior3_n2pc"),
    ("LRP",  "erpcore_lrp",        "lateral3_lrp"),
    ("ERN",  "erpcore_ern",        "midline3_ern"),
    ("N170", "erpcore_n170",       "occipital3_n170"),
    ("MMN",  "erpcore_mmn",        "midline3"),
]

# A single run of the queue.
#   extra: list of extra CLI args (for ablations)
#   tag:   ablation subdir suffix (None for main runs)
Run = namedtuple("Run", "label dataset channels model seed extra tag")


def _model_seeds(model):
    return [XDAWN_SEED] if model in CPU_ONLY_MODELS else SEEDS


def build_main_queue(channels_of):
    """channels_of: fn(ds_preset_3ch) -> actual preset ('full' or 3ch)."""
    runs = []
    models = NEURAL_MODELS + ["xdawn_rg"]
    for model in models:
        for label, ds, preset3 in DATASETS:
            ch = channels_of(preset3)
            for seed in _model_seeds(model):
                runs.append(Run(f"{label} {ch} {model} s{seed}",
                                ds, ch, model, seed, [], None))
    return runs


QUEUE_3CH = build_main_queue(lambda p3: p3)
QUEUE_FULL = build_main_queue(lambda p3: "full")


# ── Ablation grid (two-factor architecture) — R2.1 ──────────────────────
# Robustness subset kept for the native-EPMN queue (ERN / P300 / N400).
ABLATION_DATASETS = [
    ("ERN",  "erpcore_ern",  "midline3_ern"),
    ("P300", "erpcore_p300", "midline3"),
    ("N400", "erpcore_n400", "midline3_n400"),
]

# The ablation grid runs on FOUR datasets, HRI FIRST (quick/small — surfaces bad
# arms before the 40-subject sets). Base/reference = the 5-seed headline 3ch run.
ABLATION_GRID_DATASETS = [
    ("HRI",  "hri_errp_cursor", "midline3"),
    ("ERN",  "erpcore_ern",     "midline3_ern"),
    ("P300", "erpcore_p300",    "midline3"),
    ("N400", "erpcore_n400",    "midline3_n400"),
]

# (tag, model, extra CLI args). Tier-1 code arms use --model ablation_erpxttn
# (ablation_erpxttn.py); Tier-2 knob arms use --model erpxttn + a flag. Each arm
# writes to a tagged dir (erpxttn_peak_<tag> / ablation_erpxttn_<tag>) so the base
# 3ch is never overwritten. NOTE: `amp_only` and `route_only` are fusion-subset
# reads of the base checkpoints — amp_only is produced by ablation_amp_only.py
# (no training), route_only is already the base's mean_routing_auroc. Neither is a
# training run, so neither is in this queue.
ABLATION_CONFIGS = [
    # Tier 1 — the rebuild (main-text §3.4)
    ("e2e",             "ablation_erpxttn", ["--ablation-mode", "e2e"]),        # joint head, no Stage-2 fusion
    ("nowhiten",        "ablation_erpxttn", ["--ablation-mode", "nowhiten"]),   # raw cosine match
    # Tier 2 — retained knobs (supplementary)
    ("nosa",            "erpxttn", ["--no-self-attn"]),
    ("k2",              "erpxttn", ["--max-k", "2"]),
    ("k6",              "erpxttn", ["--max-k", "6"]),
    ("prom0.01",        "erpxttn", ["--peak-prominence", "0.01"]),
    ("prom0.05",        "erpxttn", ["--peak-prominence", "0.05"]),
    ("h2",              "erpxttn", ["--num-heads", "2"]),
    ("h8",              "erpxttn", ["--num-heads", "8"]),
    # Tier 3 — optional (free-head contrast; also needs the cert to make its point)
    ("learned_readout", "ablation_erpxttn", ["--ablation-mode", "learned_readout"]),
]


def build_ablation_queue():
    """HRI-first: every arm on HRI before the other three datasets."""
    runs = []
    for label, ds, preset in ABLATION_GRID_DATASETS:
        for tag, model, extra in ABLATION_CONFIGS:
            for seed in SEEDS:
                runs.append(Run(f"{label} 3ch {model}[{tag}] s{seed}",
                                ds, preset, model, seed, list(extra), tag))
    return runs


QUEUE_ABLATION = build_ablation_queue()


# ── Native-recipe EPMN robustness runs (writes to epmn_native/) ─────────
def build_epmn_native_queue():
    """EPMN under its own native recipe, on the robustness-subset datasets
    (ERN/P300/N400 — same as the ablation grid; ERN is the key imbalance test),
    both montages."""
    runs = []
    for label, ds, preset3 in ABLATION_DATASETS:
        for ch in (preset3, "full"):
            for seed in SEEDS:
                runs.append(Run(f"{label} {ch} epmn-native s{seed}",
                                ds, ch, "epmn", seed,
                                ["--epmn-recipe", "native"], None))
    return runs


QUEUE_EPMN_NATIVE = build_epmn_native_queue()


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_model_dir_name(model, tag=None):
    """Results directory name for a model (mirrors 04_train.py)."""
    name = "erpxttn_peak" if model == "erpxttn" else model
    if tag:
        name = f"{name}_{tag}"
    return name


def run_model_dir(run):
    """Results subdir for a run, accounting for the native-EPMN recipe."""
    if run.model == "epmn" and "--epmn-recipe" in run.extra and "native" in run.extra:
        return "epmn_native"
    return get_model_dir_name(run.model, run.tag)


def get_variant_dir(dataset_id, channel_preset):
    cfg_path = DATASETS_DIR / dataset_id / "dataset_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    return cfg.get("variants", {}).get(channel_preset)


def results_dir_for(run):
    variant_dir = get_variant_dir(run.dataset, run.channels)
    if not variant_dir:
        return None
    return (DATASETS_DIR / run.dataset / "results" / "tmin0ms_tmax800ms" /
            variant_dir / run_model_dir(run) / f"seed-{run.seed}")


def _parse_train_cmdline(line):
    """Parse a 04_train.py command line into a comparable key."""
    def grab(flag):
        m = re.search(rf"--{flag}\s+(\S+)", line)
        return m.group(1) if m else None
    ds, ch, model = grab("dataset"), grab("channels"), grab("model")
    if not (ds and ch and model):
        return None
    seed = grab("seed") or "42"
    tag = grab("ablation-tag")
    if model == "epmn" and grab("epmn-recipe") == "native":
        model_dir = "epmn_native"
    else:
        model_dir = get_model_dir_name(model, tag)
    return (ds, ch, model_dir, seed)


def _run_key(run):
    return (run.dataset, run.channels, run_model_dir(run), str(run.seed))


def _is_process_running(run):
    try:
        result = subprocess.run(["pgrep", "-af", "python.*04_train"],
                                capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            key = _parse_train_cmdline(line)
            if key == _run_key(run):
                return True
        return False
    except Exception:
        return False


def count_running_gpu_jobs():
    try:
        result = subprocess.run(["pgrep", "-af", "python.*04_train"],
                                capture_output=True, text=True, timeout=10)
        return sum(1 for line in result.stdout.splitlines()
                   if "xdawn_rg" not in line and "04_train" in line)
    except Exception:
        return 0


def get_run_status(run):
    """Returns (status, n_done, n_total, auroc)."""
    cfg_path = DATASETS_DIR / run.dataset / "dataset_config.json"
    if not cfg_path.exists():
        return ("no_config", 0, 0, None)
    rdir = results_dir_for(run)
    if rdir is None:
        return ("no_variant", 0, 0, None)

    results_json = rdir / "results.json"
    if results_json.exists():
        data = json.loads(results_json.read_text())
        n = len(data.get("folds", []))
        return ("done", n, n, data.get("mean_two_factor_auroc", data.get("mean_auroc")))

    n = len(list(rdir.glob("predictions_sub-*.npz")))
    cfg = json.loads(cfg_path.read_text())
    total = len(cfg.get("subjects", []))
    if n > 0:
        return ("running", n, total, None)
    if _is_process_running(run):
        return ("running", 0, total, None)
    return ("not_started", 0, total, None)


def make_log_name(run):
    short = run.dataset.replace("bnci_errp_013-2015", "bnci") \
                       .replace("hri_errp_cursor", "hri") \
                       .replace("erpcore_", "")
    parts = [short]
    if run.channels == "full":
        parts.append("full")
    parts.append(get_model_dir_name(run.model, run.tag).replace("xdawn_rg", "xdawn"))
    parts.append(f"s{run.seed}")
    return "_".join(parts) + ".log"


def launch_run(run):
    log_path = LOGS_DIR / make_log_name(run)
    cmd = [PYTHON, str(TRAIN_SCRIPT),
           "--dataset", run.dataset,
           "--channels", run.channels,
           "--model", run.model,
           "--seed", str(run.seed),
           "--resume"]
    if run.tag:
        cmd += ["--ablation-tag", run.tag]
    cmd += run.extra

    env = os.environ.copy()
    env.update(THREAD_ENV)
    log(f"LAUNCHING: {run.label} -> {log_path.name}")
    with open(log_path, "a") as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)


def run_status_check(queue):
    done, running, pending, errors = [], [], [], []
    for run in queue:
        status, n_done, n_total, auroc = get_run_status(run)
        if status == "done":
            print(f"  [DONE]    {run.label:34s}  AUROC={auroc:.4f}")
            done.append(run)
        elif status == "running":
            print(f"  [RUN]     {run.label:34s}  {n_done}/{n_total}")
            running.append(run)
        elif status == "not_started":
            pending.append(run)
        else:
            print(f"  [ERROR]   {run.label:34s}  {status}")
            errors.append(run)
    return done, running, pending, errors


def launch_pending(queue, pending):
    gpu_running = count_running_gpu_jobs()
    slots = MAX_GPU_JOBS - gpu_running
    if slots <= 0:
        log(f"No free GPU slots ({gpu_running} running, max {MAX_GPU_JOBS})")
        # CPU-only (xdawn) runs can still launch
    launched = 0
    for run in queue:
        is_cpu = run.model in CPU_ONLY_MODELS
        if not is_cpu and launched >= max(slots, 0):
            continue
        status, _, _, _ = get_run_status(run)
        if status == "not_started":
            launch_run(run)
            if not is_cpu:
                launched += 1
    return launched


def main():
    parser = argparse.ArgumentParser(description="ERP-XTTN experiment run manager")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--launch-all", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--include-full", action="store_true",
                        help="Include full-montage runs")
    parser.add_argument("--include-ablation", action="store_true",
                        help="Include the ERP-XTTN ablation grid (ERN/P300/N400, 3ch)")
    parser.add_argument("--include-epmn-native", action="store_true",
                        help="Include the native-recipe EPMN robustness runs (epmn_native/)")
    parser.add_argument("--only-epmn-native", action="store_true",
                        help="Run ONLY the native-recipe EPMN robustness runs")
    args = parser.parse_args()

    LOGS_DIR.mkdir(exist_ok=True)

    if args.only_epmn_native:
        queue = list(QUEUE_EPMN_NATIVE)
    else:
        queue = list(QUEUE_3CH)
        if args.include_full:
            queue += QUEUE_FULL
        if args.include_ablation:
            queue += QUEUE_ABLATION
        if args.include_epmn_native:
            queue += QUEUE_EPMN_NATIVE

    if args.daemon:
        log(f"Daemon started. Every {DAEMON_INTERVAL}s. MAX_GPU_JOBS={MAX_GPU_JOBS}")
        log(f"Queue: {len(queue)} runs")
        while True:
            print("\n" + "=" * 70)
            log("Status check")
            print("=" * 70)
            done, running, pending, errors = run_status_check(queue)
            gpu_running = count_running_gpu_jobs()
            log(f"Summary: {len(done)} done, {len(running)} running "
                f"({gpu_running} GPU), {len(pending)} pending")
            if not pending and not running:
                log("All runs complete!")
                break
            launched = launch_pending(queue, pending)
            if launched:
                log(f"Launched {launched} new GPU run(s)")
            time.sleep(DAEMON_INTERVAL)
        return

    print("=" * 70)
    print(f"ERP-XTTN Experiment Status  ({len(queue)} runs)")
    print("=" * 70)
    done, running, pending, errors = run_status_check(queue)
    gpu_running = count_running_gpu_jobs()
    print(f"\nSummary: {len(done)} done, {len(running)} running "
          f"({gpu_running} GPU), {len(pending)} pending")
    if errors:
        print(f"  {len(errors)} errors (missing config or variant)")

    if not (args.launch or args.launch_all):
        if pending:
            print("\nHint: use --launch to start the next batch")
        return

    launched = launch_pending(queue, pending)
    print(f"\nLaunched {launched} GPU run(s)" if launched else "\nNothing to launch")


if __name__ == "__main__":
    main()
