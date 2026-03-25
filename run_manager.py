#!/usr/bin/env python
"""
Run Manager for ERP-XTTN experiments.
Checks status of all queued runs, reports progress, and optionally launches
the next batch when current runs finish.
Usage:
    python run_manager.py              # Status check only
    python run_manager.py --launch     # Check + launch next batch if slots free
    python run_manager.py --launch-all # Keep launching until queue is full
    python run_manager.py --daemon     # Loop: check every 5min, auto-launch
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
LOGS_DIR = REPO_ROOT / "logs"
TRAIN_SCRIPT = REPO_ROOT / "04_train.py"
PYTHON = sys.executable
MAX_GPU_JOBS = 20
DAEMON_INTERVAL = 300  # seconds between daemon checks

# Thread limit for launched processes
THREAD_ENV = {
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
}

# ── Run queue ──────────────────────────────────────────────────────────
# Each entry: (label, dataset_id, channel_preset, model, rcl_weight)
# Ordered by priority. Within each model group, datasets are ordered
# fastest (fewest epochs × subjects) to slowest.

QUEUE_3CH = [
    # ── 3ch EEGNet, fastest to slowest ───────────────────────────────
    ("BNCI 3ch EEGNet",  "bnci_errp_013-2015", "midline3",          "eegnet",  0.0),
    ("HRI 3ch EEGNet",   "hri_errp_cursor",    "midline3",          "eegnet",  0.0),
    ("N400 3ch EEGNet",  "erpcore_n400",        "midline3_n400",     "eegnet",  0.0),
    ("P300 3ch EEGNet",  "erpcore_p300",        "midline3",          "eegnet",  0.0),
    ("N2pc 3ch EEGNet",  "erpcore_n2pc",        "posterior3_n2pc",   "eegnet",  0.0),
    ("LRP 3ch EEGNet",   "erpcore_lrp",         "lateral3_lrp",     "eegnet",  0.0),
    ("ERN 3ch EEGNet",   "erpcore_ern",         "midline3_ern",     "eegnet",  0.0),
    ("N170 3ch EEGNet",  "erpcore_n170",        "occipital3_n170",  "eegnet",  0.0),
    ("MMN 3ch EEGNet",   "erpcore_mmn",         "midline3",          "eegnet",  0.0),
    # ── 3ch RCL, fastest to slowest ──────────────────────────────────
    ("BNCI 3ch RCL",     "bnci_errp_013-2015",  "midline3",          "erpxttn", 0.3),
    ("HRI 3ch RCL",      "hri_errp_cursor",     "midline3",          "erpxttn", 0.3),
    ("N400 3ch RCL",     "erpcore_n400",         "midline3_n400",     "erpxttn", 0.3),
    ("P300 3ch RCL",     "erpcore_p300",         "midline3",          "erpxttn", 0.3),
    ("N2pc 3ch RCL",     "erpcore_n2pc",         "posterior3_n2pc",   "erpxttn", 0.3),
    ("LRP 3ch RCL",      "erpcore_lrp",          "lateral3_lrp",     "erpxttn", 0.3),
    ("ERN 3ch RCL",      "erpcore_ern",          "midline3_ern",     "erpxttn", 0.3),
    ("N170 3ch RCL",     "erpcore_n170",         "occipital3_n170",  "erpxttn", 0.3),
    ("MMN 3ch RCL",      "erpcore_mmn",          "midline3",          "erpxttn", 0.3),
    # ── 3ch ERPXTTN (no RCL), fastest to slowest ────────────────────
    ("BNCI 3ch ERPXTTN", "bnci_errp_013-2015",  "midline3",          "erpxttn", 0.0),
    ("HRI 3ch ERPXTTN",  "hri_errp_cursor",     "midline3",          "erpxttn", 0.0),
    ("N400 3ch ERPXTTN", "erpcore_n400",         "midline3_n400",     "erpxttn", 0.0),
    ("P300 3ch ERPXTTN", "erpcore_p300",         "midline3",          "erpxttn", 0.0),
    ("N2pc 3ch ERPXTTN", "erpcore_n2pc",         "posterior3_n2pc",   "erpxttn", 0.0),
    ("LRP 3ch ERPXTTN",  "erpcore_lrp",          "lateral3_lrp",     "erpxttn", 0.0),
    ("ERN 3ch ERPXTTN",  "erpcore_ern",          "midline3_ern",     "erpxttn", 0.0),
    ("N170 3ch ERPXTTN", "erpcore_n170",         "occipital3_n170",  "erpxttn", 0.0),
    ("MMN 3ch ERPXTTN",  "erpcore_mmn",          "midline3",          "erpxttn", 0.0),
    # ── 3ch xDAWN+RG (CPU-only, does not consume GPU slots) ─────────
    ("BNCI 3ch xDAWN",   "bnci_errp_013-2015",  "midline3",          "xdawn_rg", 0.0),
    ("HRI 3ch xDAWN",    "hri_errp_cursor",     "midline3",          "xdawn_rg", 0.0),
    ("N400 3ch xDAWN",   "erpcore_n400",         "midline3_n400",     "xdawn_rg", 0.0),
    ("P300 3ch xDAWN",   "erpcore_p300",         "midline3",          "xdawn_rg", 0.0),
    ("N2pc 3ch xDAWN",   "erpcore_n2pc",         "posterior3_n2pc",   "xdawn_rg", 0.0),
    ("LRP 3ch xDAWN",    "erpcore_lrp",          "lateral3_lrp",     "xdawn_rg", 0.0),
    ("ERN 3ch xDAWN",    "erpcore_ern",          "midline3_ern",     "xdawn_rg", 0.0),
    ("N170 3ch xDAWN",   "erpcore_n170",         "occipital3_n170",  "xdawn_rg", 0.0),
    ("MMN 3ch xDAWN",    "erpcore_mmn",          "midline3",          "xdawn_rg", 0.0),
]

QUEUE_FULL = [
    # ── Full EEGNet, fastest to slowest ──────────────────────────────
    ("BNCI full EEGNet",  "bnci_errp_013-2015", "full", "eegnet",  0.0),
    ("HRI full EEGNet",   "hri_errp_cursor",    "full", "eegnet",  0.0),
    ("N400 full EEGNet",  "erpcore_n400",        "full", "eegnet",  0.0),
    ("P300 full EEGNet",  "erpcore_p300",        "full", "eegnet",  0.0),
    ("N2pc full EEGNet",  "erpcore_n2pc",        "full", "eegnet",  0.0),
    ("LRP full EEGNet",   "erpcore_lrp",         "full", "eegnet",  0.0),
    ("ERN full EEGNet",   "erpcore_ern",         "full", "eegnet",  0.0),
    ("N170 full EEGNet",  "erpcore_n170",        "full", "eegnet",  0.0),
    ("MMN full EEGNet",   "erpcore_mmn",         "full", "eegnet",  0.0),
    # ── Full RCL, fastest to slowest ─────────────────────────────────
    ("BNCI full RCL",     "bnci_errp_013-2015",  "full", "erpxttn", 0.3),
    ("HRI full RCL",      "hri_errp_cursor",     "full", "erpxttn", 0.3),
    ("N400 full RCL",     "erpcore_n400",         "full", "erpxttn", 0.3),
    ("P300 full RCL",     "erpcore_p300",         "full", "erpxttn", 0.3),
    ("N2pc full RCL",     "erpcore_n2pc",         "full", "erpxttn", 0.3),
    ("LRP full RCL",      "erpcore_lrp",          "full", "erpxttn", 0.3),
    ("ERN full RCL",      "erpcore_ern",          "full", "erpxttn", 0.3),
    ("N170 full RCL",     "erpcore_n170",         "full", "erpxttn", 0.3),
    ("MMN full RCL",      "erpcore_mmn",          "full", "erpxttn", 0.3),
    # ── Full ERPXTTN (no RCL), fastest to slowest ───────────────────
    ("BNCI full ERPXTTN", "bnci_errp_013-2015",  "full", "erpxttn", 0.0),
    ("HRI full ERPXTTN",  "hri_errp_cursor",     "full", "erpxttn", 0.0),
    ("N400 full ERPXTTN", "erpcore_n400",         "full", "erpxttn", 0.0),
    ("P300 full ERPXTTN", "erpcore_p300",         "full", "erpxttn", 0.0),
    ("N2pc full ERPXTTN", "erpcore_n2pc",         "full", "erpxttn", 0.0),
    ("LRP full ERPXTTN",  "erpcore_lrp",          "full", "erpxttn", 0.0),
    ("ERN full ERPXTTN",  "erpcore_ern",          "full", "erpxttn", 0.0),
    ("N170 full ERPXTTN", "erpcore_n170",         "full", "erpxttn", 0.0),
    ("MMN full ERPXTTN",  "erpcore_mmn",          "full", "erpxttn", 0.0),
    # ── Full xDAWN+RG (CPU-only) ────────────────────────────────────
    ("BNCI full xDAWN",   "bnci_errp_013-2015",  "full", "xdawn_rg", 0.0),
    ("HRI full xDAWN",    "hri_errp_cursor",     "full", "xdawn_rg", 0.0),
    ("N400 full xDAWN",   "erpcore_n400",         "full", "xdawn_rg", 0.0),
    ("P300 full xDAWN",   "erpcore_p300",         "full", "xdawn_rg", 0.0),
    ("N2pc full xDAWN",   "erpcore_n2pc",         "full", "xdawn_rg", 0.0),
    ("LRP full xDAWN",    "erpcore_lrp",          "full", "xdawn_rg", 0.0),
    ("ERN full xDAWN",    "erpcore_ern",          "full", "xdawn_rg", 0.0),
    ("N170 full xDAWN",   "erpcore_n170",         "full", "xdawn_rg", 0.0),
    ("MMN full xDAWN",    "erpcore_mmn",          "full", "xdawn_rg", 0.0),
]

# Models that don't need a GPU slot
CPU_ONLY_MODELS = {"xdawn_rg"}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_model_dir_name(model, rcl_weight):
    """Determine the results directory name for a model + rcl weight."""
    if rcl_weight > 0 and model == "erpxttn":
        return f"erpxttn_rcl{rcl_weight:g}"
    return model


def get_variant_dir(dataset_id, channel_preset):
    """Look up the variant directory name from dataset_config.json."""
    cfg_path = DATASETS_DIR / dataset_id / "dataset_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    return cfg.get("variants", {}).get(channel_preset)


def _parse_train_cmdline(line):
    """Parse a 04_train.py command line into (dataset, channels, model, rcl_weight)."""
    import re
    ds = re.search(r"--dataset\s+(\S+)", line)
    ch = re.search(r"--channels\s+(\S+)", line)
    model = re.search(r"--model\s+(\S+)", line)
    rcl = re.search(r"--routing-contrast-weight\s+(\S+)", line)
    rcl_weight = float(rcl.group(1)) if rcl else 0.0
    if ds and ch and model:
        return ds.group(1), ch.group(1), model.group(1), rcl_weight
    return None, None, None, None


def _is_process_running(dataset_id, channel_preset, model_dir_name):
    """Check if a training process is already running for this combo."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "python.*04_train"],
            capture_output=True, text=True, timeout=10
        )
        # Build expected model name and rcl_weight from model_dir_name
        import re
        rcl_match = re.match(r"erpxttn_rcl(.+)", model_dir_name)
        if rcl_match:
            expected_model = "erpxttn"
            expected_rcl = float(rcl_match.group(1))
        else:
            expected_model = model_dir_name
            expected_rcl = 0.0

        for line in result.stdout.splitlines():
            ds, ch, model, rcl_weight = _parse_train_cmdline(line)
            if ds == dataset_id and ch == channel_preset \
               and model == expected_model and rcl_weight == expected_rcl:
                return True
        return False
    except Exception:
        return False


def count_running_gpu_jobs():
    """Count running 04_train.py GPU processes (exclude xdawn_rg)."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "python.*04_train"],
            capture_output=True, text=True, timeout=10
        )
        count = 0
        for line in result.stdout.splitlines():
            if "xdawn_rg" not in line and "04_train" in line:
                count += 1
        return count
    except Exception:
        return 0


def get_run_status(dataset_id, channel_preset, model, rcl_weight):
    """
    Returns (status, n_done, n_total, auroc).
    status: 'done' | 'running' | 'not_started' | 'no_config' | 'no_variant'
    """
    cfg_path = DATASETS_DIR / dataset_id / "dataset_config.json"
    if not cfg_path.exists():
        return ("no_config", 0, 0, None)

    variant_dir = get_variant_dir(dataset_id, channel_preset)
    if not variant_dir:
        return ("no_variant", 0, 0, None)

    model_dir = get_model_dir_name(model, rcl_weight)
    results_path = (DATASETS_DIR / dataset_id / "results" /
                    "tmin0ms_tmax800ms" / variant_dir / model_dir)
    results_json = results_path / "results.json"

    if results_json.exists():
        data = json.loads(results_json.read_text())
        n = len(data.get("folds", []))
        return ("done", n, n, data.get("mean_auroc"))

    preds = list(results_path.glob("predictions_sub-*.npz"))
    n = len(preds)

    cfg = json.loads(cfg_path.read_text())
    total = len(cfg.get("subjects", []))

    if n > 0:
        return ("running", n, total, None)

    if _is_process_running(dataset_id, channel_preset, model_dir):
        return ("running", 0, total, None)

    return ("not_started", 0, total, None)


def make_log_name(dataset_id, channel_preset, model_dir):
    """Build a log filename from run parameters."""
    # Shorten dataset name
    short = dataset_id.replace("bnci_errp_013-2015", "bnci") \
                       .replace("hri_errp_cursor", "hri") \
                       .replace("erpcore_", "")
    variant = "full" if channel_preset == "full" else ""
    model_short = model_dir.replace("erpxttn_rcl0.3", "rcl") \
                           .replace("xdawn_rg", "xdawn")
    parts = [short]
    if variant:
        parts.append(variant)
    parts.append(model_short)
    return "_".join(parts) + ".log"


def launch_run(dataset_id, channel_preset, model, rcl_weight):
    """Launch a training run as a background process with thread limits."""
    model_dir = get_model_dir_name(model, rcl_weight)
    log_name = make_log_name(dataset_id, channel_preset, model_dir)
    log_path = LOGS_DIR / log_name

    cmd = [PYTHON, str(TRAIN_SCRIPT),
           "--dataset", dataset_id,
           "--channels", channel_preset,
           "--model", model,
           "--resume"]
    if rcl_weight > 0:
        cmd += ["--routing-contrast-weight", str(rcl_weight)]

    env = os.environ.copy()
    env.update(THREAD_ENV)

    log(f"LAUNCHING: {dataset_id} / {channel_preset} / {model_dir} -> {log_name}")
    with open(log_path, "a") as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)


GEN_FIGURES_SCRIPT = REPO_ROOT / "05_gen_figures.py"


def generate_figures_if_needed(dataset_id, channel_preset, model, rcl_weight):
    """Generate figures for a completed ERPXTTN-family run if not already done."""
    if model not in ("erpxttn",):
        return
    model_dir = get_model_dir_name(model, rcl_weight)
    variant_dir = get_variant_dir(dataset_id, channel_preset)
    if not variant_dir:
        return

    results_path = (DATASETS_DIR / dataset_id / "results" /
                    "tmin0ms_tmax800ms" / variant_dir / model_dir)

    existing_figs = list(results_path.glob("fig_*.png"))
    if len(existing_figs) >= 5:
        return

    if not (results_path / "results.json").exists():
        return

    env = os.environ.copy()
    env.update(THREAD_ENV)

    cmd = [PYTHON, str(GEN_FIGURES_SCRIPT),
           "--dataset", dataset_id,
           "--channels", channel_preset,
           "--model", model_dir]
    log(f"GENERATING FIGURES: {dataset_id} / {channel_preset} / {model_dir}")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def run_status_check(queue):
    """Print status and return categorized runs."""
    done_runs = []
    running_runs = []
    pending_runs = []
    error_runs = []

    for label, ds, ch, model, rcl in queue:
        status, n_done, n_total, auroc = get_run_status(ds, ch, model, rcl)
        if status == "done":
            print(f"  [DONE]    {label:30s}  AUROC={auroc:.4f}")
            done_runs.append((label, ds, ch, model, rcl))
        elif status == "running":
            print(f"  [RUN]     {label:30s}  {n_done}/{n_total}")
            running_runs.append((label, ds, ch, model, rcl))
        elif status == "not_started":
            print(f"  [PENDING] {label:30s}")
            pending_runs.append((label, ds, ch, model, rcl))
        else:
            print(f"  [ERROR]   {label:30s}  {status}")
            error_runs.append((label, ds, ch, model, rcl))

    return done_runs, running_runs, pending_runs, error_runs


def launch_pending(queue, done_runs, running_runs, pending_runs):
    """Launch pending runs up to MAX_GPU_JOBS. Returns number launched."""
    gpu_running = count_running_gpu_jobs()
    slots = MAX_GPU_JOBS - gpu_running

    if slots <= 0:
        log(f"No free GPU slots ({gpu_running} GPU jobs running, max {MAX_GPU_JOBS})")
        return 0

    # Generate figures for completed ERPXTTN-family runs
    for label, ds, ch, model, rcl in done_runs:
        generate_figures_if_needed(ds, ch, model, rcl)

    launched = 0
    for label, ds, ch, model, rcl in queue:
        if model not in CPU_ONLY_MODELS and launched >= slots:
            break
        status, _, _, _ = get_run_status(ds, ch, model, rcl)
        if status == "not_started":
            launch_run(ds, ch, model, rcl)
            if model not in CPU_ONLY_MODELS:
                launched += 1

    return launched


def main():
    parser = argparse.ArgumentParser(description="ERP-XTTN experiment run manager")
    parser.add_argument("--launch", action="store_true",
                        help="Launch next batch of runs if GPU slots are available")
    parser.add_argument("--launch-all", action="store_true",
                        help="Keep launching until queue is full")
    parser.add_argument("--daemon", action="store_true",
                        help="Loop forever: check every 5min, auto-launch pending runs")
    parser.add_argument("--include-full", action="store_true",
                        help="Include full-channel runs in the queue")
    args = parser.parse_args()

    LOGS_DIR.mkdir(exist_ok=True)

    queue = list(QUEUE_3CH)
    if args.include_full:
        queue += QUEUE_FULL

    if args.daemon:
        log(f"Daemon started. Checking every {DAEMON_INTERVAL}s. MAX_GPU_JOBS={MAX_GPU_JOBS}")
        log(f"Queue: {len(queue)} runs ({'3ch + full' if args.include_full else '3ch only'})")
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

            launched = launch_pending(queue, done, running, pending)
            if launched:
                log(f"Launched {launched} new run(s)")

            time.sleep(DAEMON_INTERVAL)
        return

    # ── One-shot status report ─────────────────────────────────────────
    print("=" * 70)
    print("ERP-XTTN Experiment Status")
    print("=" * 70)

    done, running, pending, errors = run_status_check(queue)

    gpu_running = count_running_gpu_jobs()
    cpu_running = [r for r in running if r[3] in CPU_ONLY_MODELS]

    print(f"\nSummary: {len(done)} done, {len(running)} running "
          f"({gpu_running} GPU + {len(cpu_running)} CPU), "
          f"{len(pending)} pending")
    if errors:
        print(f"  {len(errors)} errors (missing config or variant)")

    if not (args.launch or args.launch_all):
        if pending and not running:
            print("\nHint: use --launch to start the next batch")
        return

    launched = launch_pending(queue, done, running, pending)
    if launched:
        print(f"\nLaunched {launched} GPU run(s)")
    else:
        print("\nNothing to launch")


if __name__ == "__main__":
    main()
