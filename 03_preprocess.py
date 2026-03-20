"""Unified preprocessing + epoching pipeline.

Merges the logic from the original preprocessing, epoching, and channel
selection steps into a single unified script.

Pipeline per raw FIF file:
  1. Load raw FIF from {dataset}/raw_fif/{sub}/{ses}/*.fif
  2. Drop QC-flagged channels (from qc_channel_stats.csv if it exists)
  3. Apply reference: "car" -> common average, "none" -> skip
  4. Pick channel subset (CHANNEL_PRESETS, or "full")
  5. IIR 1-10 Hz bandpass (forward-phase Butterworth 4th order)
  6. Optional resample
  7. Epoch (event_lock or trial_outcome, from dataset_config.json)
  8. Save epoch FIF

Variant naming:
  {ref}[_{n}ch][_rs{freq}]_{filter}_bp-1-10

  Full channels omit the channel count (dataset-agnostic):
    car_iir_fwd_bp-1-10             car_fir_zero_bp-1-10
  Explicit subsets include it:
    noref_2ch_iir_fwd_bp-1-10       noref_3ch_fir_zero_bp-1-10

Usage:
  python 03_preprocess.py --dataset datasets/bnci_errp_013-2015 --channels full --reference car
  python 03_preprocess.py --dataset datasets/bnci_errp_013-2015 --channels full --reference car --filter-method fir_zero
  python 03_preprocess.py --dataset datasets/hri_errp_cursor --channels midline3 --reference none
  python 03_preprocess.py --dataset datasets/bnci_errp_013-2015 --channels midline2_bnci --reference none --resample 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Channel presets
# ---------------------------------------------------------------------------
CHANNEL_PRESETS = {
    # Generic cross-dataset presets
    "midline2": ["Fz", "Cz"],
    "midline3": ["Fz", "Cz", "Pz"],
    # Dataset-specific presets should be defined in dataset_config.json
    # under "channel_presets". They override these built-in presets.
}


# ---------------------------------------------------------------------------
# Filter parameters
# ---------------------------------------------------------------------------
DEFAULT_L_FREQ = 1.0
DEFAULT_H_FREQ = 10.0

FILTER_METHODS = {
    "iir_fwd": {
        "method": "iir",
        "iir_params": {"order": 4, "ftype": "butter"},
        "phase": "forward",
        "label": "IIR fwd Butterworth 4th order",
        "variant_tag": "iir_fwd",
    },
    "fir_zero": {
        "method": "fir",
        "phase": "zero",
        "label": "FIR zero-phase (firwin)",
        "variant_tag": "fir_zero",
    },
}

# ---------------------------------------------------------------------------
# Epoch time-window variants
# ---------------------------------------------------------------------------
EPOCH_VARIANTS = {
    "tmin-200ms_tmax600ms": {"tmin": -0.200, "tmax": 0.600, "baseline": (-0.2, 0)},
    "tmin0ms_tmax800ms":    {"tmin":  0.000, "tmax": 0.800, "baseline": None},
}


# ---------------------------------------------------------------------------
# BIDS file discovery
# ---------------------------------------------------------------------------
def iter_raw_fif_files(
    root: Path,
    task: str,
    only_subject: str | None = None,
    only_session: str | None = None,
) -> list[Path]:
    """Find raw FIF files matching BIDS naming under *root*."""
    pattern = f"**/sub-*_ses-*_task-{task}_run-*_raw.fif"
    found = sorted(root.glob(pattern))
    if only_subject:
        found = [p for p in found if f"sub-{only_subject}" in str(p)]
    if only_session:
        found = [p for p in found if f"ses-{only_session}" in str(p)]
    return found


def parse_bids_bits(fif_path: Path) -> tuple[str, str, str]:
    """Extract (sub-XX, ses-YY, base) from a raw FIF path."""
    base = fif_path.name.replace("_raw.fif", "")
    parts = fif_path.parts
    sub = next((x for x in parts if x.startswith("sub-")), "sub-unk")
    ses = next((x for x in parts if x.startswith("ses-")), "ses-unk")
    return sub, ses, base


# ---------------------------------------------------------------------------
# QC channel flagging
# ---------------------------------------------------------------------------
def load_flagged_channels(dataset_dir: Path) -> dict[str, list[str]]:
    """Load QA-flagged channels from ``qc_outputs/qc_channel_stats.csv``.

    Returns a mapping of ``{fif_base: [flagged_ch_name, ...]}``.
    If the CSV does not exist or has unexpected columns, returns an empty dict.
    """
    qc_csv = dataset_dir / "qc_outputs" / "qc_channel_stats.csv"
    flagged_map: dict[str, list[str]] = {}
    if qc_csv.exists():
        df = pd.read_csv(qc_csv)
        required = {"fif_base", "ch_name", "any_flag"}
        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] QC CSV missing columns {missing}; skipping channel drop.")
            return flagged_map
        df_flag = df[df["any_flag"] == True].copy()  # noqa: E712
        flagged_map = {
            base: g["ch_name"].astype(str).tolist()
            for base, g in df_flag.groupby("fif_base")
        }
    else:
        print(f"[WARN] No QC CSV at {qc_csv}; no channels will be dropped.")
    return flagged_map


# ---------------------------------------------------------------------------
# Variant naming
# ---------------------------------------------------------------------------
def make_variant_name(
    reference: str,
    preset: str,
    resample_freq: float | None,
    filter_tag: str,
    qc_drop: bool = True,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
) -> str:
    """Build the preprocessing variant directory name.

    Uses the preset name directly (e.g. ``midline2``, ``midline2_bnci``)
    so that presets with the same channel count but different channels
    produce distinct directory names.

    Examples:
      car_qcdrop_iir_fwd_bp-1-10                (full channels, CAR, QC drop, IIR)
      noref_midline2_rs256_iir_fwd_bp-1-10      (midline2, no ref, no QC drop, IIR)
      noref_qcdrop_rs256_fir_zero_bp-1-10       (full, no ref, QC drop, FIR)
    """
    ref = "car" if reference == "car" else "noref"
    ch = "" if preset == "full" else f"_{preset}"
    qc = "_qcdrop" if qc_drop else ""
    rs = f"_rs{int(resample_freq)}" if resample_freq else ""
    l_str = int(l_freq) if l_freq == int(l_freq) else l_freq
    h_str = int(h_freq) if h_freq == int(h_freq) else h_freq
    return f"{ref}{ch}{qc}{rs}_{filter_tag}_bp-{l_str}-{h_str}"


# ---------------------------------------------------------------------------
# Epoching: event_lock mode (BNCI-style)
# ---------------------------------------------------------------------------
def epoch_event_lock(
    raw: mne.io.BaseRaw,
    event_subset: list[str] | None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract events from annotations for event_lock epoching.

    If *event_subset* is None, all annotation events are used.
    Otherwise only annotations whose description is in the subset are kept.
    """
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")

    if event_subset is not None:
        subset_set = set(event_subset)
        keep_ids = {
            name: code for name, code in event_id.items() if name in subset_set
        }
        if not keep_ids:
            print(f"    [WARN] No events matched subset {event_subset}")
            return np.zeros((0, 3), dtype=int), {}
        keep_codes = set(keep_ids.values())
        mask = np.isin(events[:, 2], list(keep_codes))
        events = events[mask]
        event_id = keep_ids

    for name, code in sorted(event_id.items(), key=lambda x: x[1]):
        count = int(np.sum(events[:, 2] == code)) if len(events) > 0 else 0
        print(f"    {name} (code={code}): {count} events")

    return events, event_id


# ---------------------------------------------------------------------------
# Epoching: trial_outcome mode (HRI-style)
# ---------------------------------------------------------------------------
def build_trial_outcome_events(
    raw: mne.io.BaseRaw,
    time_lock_names: list[str],
    outcome_map: dict[str, str],
    exclude_outcomes: list[str],
    search_window_sec: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build composite events for trial_outcome epoching.

    For each time-lock event (e.g. feedback markers), search forward for the
    next outcome event (correct / error) within *search_window_sec* seconds.
    """
    sfreq = raw.info["sfreq"]

    # Reverse map: annotation_description -> label_name
    outcome_desc_to_label = {desc: label for label, desc in outcome_map.items()}
    all_outcome_descs = set(outcome_desc_to_label.keys()) | set(exclude_outcomes)

    annot = raw.annotations
    onsets = annot.onset
    descriptions = annot.description

    event_id = {label: i + 1 for i, label in enumerate(outcome_map.keys())}

    time_lock_set = set(time_lock_names)
    events_list: list[list[int]] = []
    n_excluded = 0
    n_no_outcome = 0

    for i, desc in enumerate(descriptions):
        if desc not in time_lock_set:
            continue
        onset_sec = onsets[i]
        sample = int(round(onset_sec * sfreq))

        found_outcome = None
        for j in range(i + 1, len(descriptions)):
            dt = onsets[j] - onset_sec
            if dt > search_window_sec:
                break
            if descriptions[j] in all_outcome_descs:
                found_outcome = descriptions[j]
                break

        if found_outcome is None:
            n_no_outcome += 1
            continue
        if found_outcome in exclude_outcomes:
            n_excluded += 1
            continue

        label = outcome_desc_to_label.get(found_outcome)
        if label is None:
            continue

        events_list.append([sample, 0, event_id[label]])

    if n_excluded > 0:
        print(f"    Excluded {n_excluded} trials (outcomes: {exclude_outcomes})")
    if n_no_outcome > 0:
        print(
            f"    [WARN] {n_no_outcome} time-lock events had no outcome "
            f"within {search_window_sec}s window"
        )

    events = (
        np.array(events_list, dtype=int)
        if events_list
        else np.zeros((0, 3), dtype=int)
    )

    for label, code in event_id.items():
        count = int(np.sum(events[:, 2] == code)) if len(events) > 0 else 0
        print(f"    {label} (code={code}): {count} trials")

    return events, event_id


# ---------------------------------------------------------------------------
# Core: process one raw FIF file
# ---------------------------------------------------------------------------
def process_one_file(
    fif_path: Path,
    flagged_map: dict[str, list[str]],
    epochs_root: Path,
    epoch_cfg: dict,
    epoch_variant_name: str,
    epoch_variant_params: dict,
    variant_name: str,
    channel_list: list[str] | None,
    reference: str,
    resample_freq: float | None,
    filter_params: dict,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
) -> None:
    """Load raw FIF, apply QC drops + reference + channel pick + filter +
    resample + epoch, and save the result.
    """
    sub, ses, base = parse_bids_bits(fif_path)
    print(f"\n=== {base} ===")

    # 1. Load raw data
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    print(
        f"  Loaded: {len(raw.ch_names)} channels, "
        f"{raw.n_times} samples, {raw.info['sfreq']} Hz"
    )

    # 2. Drop QC-flagged channels
    flagged = flagged_map.get(base, [])
    if flagged:
        to_drop = [ch for ch in flagged if ch in raw.ch_names]
        if to_drop:
            raw.drop_channels(to_drop)
            print(f"  Dropped flagged: {to_drop}")
    else:
        print(f"  No flagged channels to drop.")

    # 3. Apply reference
    if reference == "car":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        print(f"  Applied CAR (common average reference)")
    else:
        print(f"  No re-referencing (keeping original recording reference)")

    # 4. Pick channel subset (intersect with what actually exists)
    if channel_list is not None:
        available = [ch for ch in channel_list if ch in raw.ch_names]
        missing = [ch for ch in channel_list if ch not in raw.ch_names]
        if missing:
            print(f"  [WARN] Missing channels: {missing}")
        if len(available) < 1:
            print(f"  [SKIP] No requested channels available.")
            return
        raw.pick(available)
        print(f"  Picked {len(available)} channels: {available}")
    else:
        # "full" mode: pick only EEG channels (drop EOG, misc, stim, etc.)
        eeg_picks = mne.pick_types(
            raw.info, eeg=True, meg=False, eog=False, stim=False, misc=False
        )
        eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]
        raw.pick(eeg_ch_names)
        print(f"  Picked all {len(eeg_ch_names)} EEG channels")

    # 5. Bandpass filter
    filt_kwargs = {"l_freq": l_freq, "h_freq": h_freq, "verbose": "ERROR",
                   "method": filter_params["method"], "phase": filter_params["phase"]}
    if "iir_params" in filter_params:
        filt_kwargs["iir_params"] = filter_params["iir_params"]
    raw.filter(**filt_kwargs)
    print(f"  Filtered: {filter_params['label']}, {l_freq}-{h_freq} Hz")

    # 6. Resample (optional)
    if resample_freq and raw.info["sfreq"] != resample_freq:
        orig_sfreq = raw.info["sfreq"]
        raw.resample(resample_freq, npad="auto", verbose="ERROR")
        print(f"  Resampled: {orig_sfreq} -> {resample_freq} Hz")
    elif resample_freq:
        print(f"  Already at {resample_freq} Hz, no resampling needed")

    # 7. Extract events
    epoch_mode = epoch_cfg.get("mode", "event_lock")

    if epoch_mode == "trial_outcome":
        events, event_id = build_trial_outcome_events(
            raw,
            time_lock_names=epoch_cfg["time_lock_events"],
            outcome_map=epoch_cfg["outcome_events"],
            exclude_outcomes=epoch_cfg.get("exclude_outcomes", []),
            search_window_sec=epoch_cfg.get("outcome_search_window_sec", 2.0),
        )
    elif epoch_mode == "event_lock":
        events, event_id = epoch_event_lock(
            raw,
            event_subset=epoch_cfg.get("event_subset"),
        )
    else:
        raise ValueError(f"Unknown epoching mode: {epoch_mode!r}")

    if len(events) == 0:
        print(f"  [WARN] No events found, skipping.")
        return

    # 8. Create epochs and save
    print(
        f"  Epoching: {epoch_variant_name} "
        f"(tmin={epoch_variant_params['tmin']}, tmax={epoch_variant_params['tmax']})"
    )
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=epoch_variant_params["tmin"],
        tmax=epoch_variant_params["tmax"],
        baseline=epoch_variant_params["baseline"],
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    out_dir = epochs_root / epoch_variant_name / variant_name / sub / ses
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{base}-epo.fif"
    epochs.save(out_path, overwrite=True, verbose="ERROR")
    print(
        f"  Saved: {out_path}\n"
        f"    {len(epochs)} epochs, {len(epochs.ch_names)} channels, "
        f"sfreq={epochs.info['sfreq']} Hz"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified preprocessing + epoching pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python 03_preprocess.py --dataset datasets/bnci_errp_013-2015 --channels full --reference car
  python 03_preprocess.py --dataset datasets/hri_errp_cursor --channels midline3 --reference none
  python 03_preprocess.py --dataset datasets/bnci_errp_013-2015 --channels midline2_bnci --reference none --resample 256
""",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset directory (must contain dataset_config.json and raw_fif/).",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="full",
        help=(
            'Channel preset name (from CHANNEL_PRESETS keys, or "full"). '
            f"Available presets: {sorted(CHANNEL_PRESETS.keys())}. Default: full."
        ),
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="none",
        choices=["car", "none"],
        help='Reference method: "car" (common average) or "none" (no re-referencing). Default: none.',
    )
    parser.add_argument(
        "--filter-method",
        type=str,
        default="iir_fwd",
        choices=list(FILTER_METHODS.keys()),
        help=f"Filter method: {list(FILTER_METHODS.keys())}. Default: iir_fwd.",
    )
    parser.add_argument(
        "--l-freq",
        type=float,
        default=DEFAULT_L_FREQ,
        help=f"Low cutoff frequency in Hz (default: {DEFAULT_L_FREQ}).",
    )
    parser.add_argument(
        "--h-freq",
        type=float,
        default=DEFAULT_H_FREQ,
        help=f"High cutoff frequency in Hz (default: {DEFAULT_H_FREQ}).",
    )
    parser.add_argument(
        "--resample",
        type=float,
        default=None,
        help="Target sampling rate in Hz (e.g. 256). Default: no resampling.",
    )
    parser.add_argument(
        "--epoch-variant",
        type=str,
        default="tmin0ms_tmax800ms",
        choices=list(EPOCH_VARIANTS.keys()),
        help=f"Epoch time window. Choices: {list(EPOCH_VARIANTS.keys())}. Default: tmin0ms_tmax800ms.",
    )
    parser.add_argument(
        "--no-qc-drop",
        action="store_true",
        default=False,
        help="Skip dropping QC-flagged channels (keep all channels).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Only process this subject (e.g. '01').",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Only process this session (e.g. '01').",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve dataset directory and load config
    # ------------------------------------------------------------------
    dataset_dir = args.dataset.resolve()
    cfg_path = dataset_dir / "dataset_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No dataset_config.json found in {dataset_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    task = cfg.get("bids_task", "errp")
    epoch_cfg = cfg.get("epoching", {"mode": "event_lock", "event_subset": None})

    # ------------------------------------------------------------------
    # Resolve channel list (dataset config overrides built-in presets)
    # ------------------------------------------------------------------
    dataset_channel_presets = cfg.get("channel_presets", {})
    all_presets = {**CHANNEL_PRESETS, **dataset_channel_presets}
    if args.channels == "full":
        channel_list = None  # sentinel: pick all EEG channels per file
    elif args.channels in all_presets:
        channel_list = all_presets[args.channels]
    else:
        raise ValueError(
            f"Unknown channel preset: {args.channels!r}. "
            f"Use 'full' or one of {sorted(all_presets.keys())}."
        )

    # ------------------------------------------------------------------
    # To compute variant name we need the channel count.
    # For "full" we peek at the first raw file to count EEG channels.
    # ------------------------------------------------------------------
    raw_fif_root = dataset_dir / "raw_fif"
    if not raw_fif_root.exists():
        raise FileNotFoundError(f"No raw_fif directory at {raw_fif_root}")

    fif_files = iter_raw_fif_files(
        raw_fif_root, task,
        only_subject=args.subject,
        only_session=args.session,
    )
    if not fif_files:
        raise FileNotFoundError(f"No raw FIF files found under {raw_fif_root}")

    # Load QC flags (may reduce channel count), unless --no-qc-drop
    if args.no_qc_drop:
        flagged_map = {}
        print("QC channel drop:   DISABLED (--no-qc-drop)")
    else:
        flagged_map = load_flagged_channels(dataset_dir)

    is_full = (args.channels == "full")
    filter_cfg = FILTER_METHODS[args.filter_method]

    if is_full:
        n_channels = None  # not used in variant name for "full"
    else:
        n_channels = len(channel_list) if channel_list else 0

    l_freq = args.l_freq
    h_freq = args.h_freq

    variant_name = make_variant_name(
        args.reference, args.channels, args.resample,
        filter_tag=filter_cfg["variant_tag"],
        qc_drop=not args.no_qc_drop,
        l_freq=l_freq, h_freq=h_freq,
    )

    # Epoch variant
    epoch_variant_name = args.epoch_variant
    epoch_variant_params = EPOCH_VARIANTS[epoch_variant_name]

    epochs_root = dataset_dir / "epoched_fif"

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"Dataset:           {cfg.get('dataset_name', dataset_dir.name)}")
    print(f"Task:              {task}")
    print(f"Channels:          {args.channels}" + (f" ({n_channels} ch)" if n_channels else " (full)"))
    if channel_list is not None:
        print(f"  Preset:          {channel_list}")
    print(f"Reference:         {args.reference}")
    print(f"Filter:            {filter_cfg['label']}, {l_freq}-{h_freq} Hz")
    print(f"Resample:          {args.resample or 'None (keep original)'}")
    print(f"Epoch variant:     {epoch_variant_name}")
    print(f"Epoching mode:     {epoch_cfg.get('mode', 'event_lock')}")
    print(f"Variant name:      {variant_name}")
    print(f"Output root:       {epochs_root}")
    print(f"Files to process:  {len(fif_files)}")

    if epoch_cfg.get("mode") == "trial_outcome":
        print(f"  Time-lock:       {epoch_cfg['time_lock_events']}")
        print(f"  Outcomes:        {epoch_cfg['outcome_events']}")
        print(f"  Exclude:         {epoch_cfg.get('exclude_outcomes', [])}")
        print(
            f"  Search window:   {epoch_cfg.get('outcome_search_window_sec', 2.0)}s"
        )

    # ------------------------------------------------------------------
    # Process each file
    # ------------------------------------------------------------------
    for fif_path in fif_files:
        process_one_file(
            fif_path=fif_path,
            flagged_map=flagged_map,
            epochs_root=epochs_root,
            epoch_cfg=epoch_cfg,
            epoch_variant_name=epoch_variant_name,
            epoch_variant_params=epoch_variant_params,
            variant_name=variant_name,
            channel_list=channel_list,
            reference=args.reference,
            resample_freq=args.resample,
            filter_params=filter_cfg,
            l_freq=l_freq,
            h_freq=h_freq,
        )

    print(f"\nPreprocessing + epoching complete.")
    print(f"Output: {epochs_root / epoch_variant_name / variant_name}")


if __name__ == "__main__":
    main()
