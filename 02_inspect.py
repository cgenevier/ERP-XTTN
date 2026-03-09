from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

# -------------------------
# Config defaults
# -------------------------

# Plot config (saved to disk; no interactive popups)
SAVE_RAW_SNIPPET_PLOT = True
SAVE_PSD_PLOT = True
SAVE_EVENTS_PLOT = True

RAW_SNIPPET_SEC = 10.0  # how much raw to plot (from start of recording)
PSD_FMIN = 0.1
PSD_FMAX = 120.0

# QC windows / sampling control
QC_WINDOW_SEC = 60.0
MAX_SAMPLES_FOR_STATS = 50_000

# Thresholds for dead channel detection
DEAD_STD_RATIO = 0.05
DEAD_PTP_RATIO = 0.05
DEAD_MAD_RATIO = 0.05
NEAR_ZERO_VAR = 1e-18

# Thresholds for basic channel QC (in Volts)
STD_MIN_V = 1e-9
STD_MAX_V = 5e-3
PTP_MAX_V = 2e-2
MAD_MAX_V = 2e-3

# Correlation thresholds
MEAN_ABS_CORR_MIN = 0.05
MEAN_ABS_CORR_MAX = 0.98

# -------------------------
# Helpers
# -------------------------
def iter_fif_files(
    root: Path, task: str,
    only_subject: str | None = None,
    only_session: str | None = None,
    only_run: str | None = None,
) -> list[Path]:
    pattern = f"**/sub-*_ses-*_task-{task}_run-*_raw.fif"
    found = sorted(root.glob(pattern))
    if only_subject:
        found = [p for p in found if f"sub-{only_subject}" in str(p)]
    if only_session:
        found = [p for p in found if f"ses-{only_session}" in str(p)]
    if only_run:
        found = [p for p in found if f"run-{only_run}" in p.name]
    return found


def find_sidecars(fif_path: Path):
    base = fif_path.name.replace("_raw.fif", "")
    events_csv = fif_path.parent / f"{base}_events.csv"
    meta_json = fif_path.parent / f"{base}_meta.json"
    return events_csv, meta_json, base


def load_events_csv(events_csv: Path) -> pd.DataFrame:
    if not events_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(events_csv)


def load_meta_json(meta_json: Path) -> dict:
    if not meta_json.exists():
        return {}
    with open(meta_json, "r") as f:
        return json.load(f)


def events_df_to_mne_events(events_df: pd.DataFrame, sample_col: str = "sample0") -> np.ndarray:
    if events_df is None or len(events_df) == 0:
        return np.zeros((0, 3), dtype=int)
    return np.column_stack(
        [
            events_df[sample_col].astype(int).to_numpy(),
            np.zeros(len(events_df), dtype=int),
            events_df["code"].astype(int).to_numpy(),
        ]
    ).astype(int)


def _select_qc_segment(raw: mne.io.BaseRaw, window_sec: float | None) -> tuple[int, int]:
    """Return (start, stop) sample indices for QC segment."""
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    start = 0
    if window_sec is None:
        stop = n_times
    else:
        stop = min(n_times, int(round(window_sec * sfreq)))
        stop = max(stop, 1)
    return start, stop


def _decimation_factor(n_samples: int, max_samples: int) -> int:
    """Compute integer decimation factor so that n_samples/decim <= max_samples."""
    if n_samples <= max_samples:
        return 1
    return int(np.ceil(n_samples / max_samples))


def compute_channel_stats(
    raw: mne.io.BaseRaw,
    picks: np.ndarray,
    window_sec: float | None,
    max_samples: int
) -> pd.DataFrame:
    """
    Compute per-channel stats on a segment of raw data using decimated samples:
      - mean, std, var
      - min, max, peak-to-peak
      - median, MAD (median absolute deviation)
    Returns a DataFrame indexed by channel name.
    """
    start, stop = _select_qc_segment(raw, window_sec)
    n_samples = stop - start
    decim = _decimation_factor(n_samples, max_samples)

    data = raw.get_data(picks=picks, start=start, stop=stop, reject_by_annotation="omit")
    # decimate in time to cap sample count
    data = data[:, ::decim]  # shape: (n_ch, n_t)

    ch_names = [raw.ch_names[p] for p in picks]

    mean = data.mean(axis=1)
    var = data.var(axis=1, ddof=0)
    std = np.sqrt(var)
    mn = data.min(axis=1)
    mx = data.max(axis=1)
    ptp = mx - mn

    med = np.median(data, axis=1)
    mad = np.median(np.abs(data - med[:, None]), axis=1)

    df = pd.DataFrame(
        {
            "ch_name": ch_names,
            "qc_window_sec": float(window_sec) if window_sec is not None else np.nan,
            "qc_samples_used": data.shape[1],
            "decim": decim,
            "mean_v": mean,
            "var_v2": var,
            "std_v": std,
            "min_v": mn,
            "max_v": mx,
            "ptp_v": ptp,
            "median_v": med,
            "mad_v": mad,
        }
    )
    return df


def compute_mean_abs_correlation(
    raw: mne.io.BaseRaw,
    picks: np.ndarray,
    window_sec: float | None,
    max_samples: int
) -> pd.DataFrame:
    """
    Compute per-channel mean absolute correlation with all other channels
    (on decimated data). Flags suspiciously low/high mean abs corr.
    """
    start, stop = _select_qc_segment(raw, window_sec)
    n_samples = stop - start
    decim = _decimation_factor(n_samples, max_samples)

    data = raw.get_data(picks=picks, start=start, stop=stop, reject_by_annotation="omit")
    data = data[:, ::decim]  # (n_ch, n_t)

    # Demean each channel to reduce correlation inflation from DC offsets
    data = data - data.mean(axis=1, keepdims=True)

    # Corrcoef expects rows as variables if rowvar=True; our rows are channels
    C = np.corrcoef(data)
    # Replace NaNs (can occur if a channel is flat) with 0
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    n_ch = C.shape[0]
    mean_abs_corr = []
    for i in range(n_ch):
        others = np.abs(np.delete(C[i, :], i))
        mean_abs_corr.append(float(np.mean(others)) if others.size else 0.0)

    ch_names = [raw.ch_names[p] for p in picks]
    df = pd.DataFrame(
        {
            "ch_name": ch_names,
            "qc_window_sec": float(window_sec) if window_sec is not None else np.nan,
            "qc_samples_used": data.shape[1],
            "decim": decim,
            "mean_abs_corr": mean_abs_corr,
        }
    )
    return df


def save_raw_snippet_plot(raw: mne.io.BaseRaw, fif_base: str, out_dir: Path, duration_sec: float, max_channels: int = 20):
    """
    Save a static multi-channel raw snippet plot (matplotlib).
    Fix: remove per-channel DC offset before scaling so lines aren't flat.
    """
    sfreq = float(raw.info["sfreq"])
    stop = min(int(raw.n_times), int(round(duration_sec * sfreq)))
    stop = max(stop, 1)

    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, misc=False)
    if max_channels is not None:
        picks = picks[: min(max_channels, len(picks))]

    data = raw.get_data(picks=picks, start=0, stop=stop)
    times = np.arange(data.shape[1]) / sfreq

    # Remove per-channel DC offset (median is robust)
    data_centered = data - np.median(data, axis=1, keepdims=True)

    # Robust scale using MAD across all plotted data
    med = np.median(data_centered)
    mad = np.median(np.abs(data_centered - med))
    scale = 6 * mad  # ~4-6*MAD gives a nice visible scale
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(data_centered) if np.std(data_centered) > 0 else 1e-6

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    offsets = np.arange(len(picks))[::-1]
    for i, p in enumerate(picks):
        ax.plot(times, data_centered[i] / scale + offsets[i], linewidth=0.7)

    ax.set_yticks(offsets)
    ax.set_yticklabels([raw.ch_names[p] for p in picks])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{fif_base}: Raw EEG snippet (first {duration_sec:.1f}s, median-centered, scaled)")
    ax.grid(True, alpha=0.2)

    out_path = out_dir / f"{fif_base}_raw_snippet_{int(duration_sec)}s.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_raw_overview_plot(
    raw: mne.io.BaseRaw,
    fif_base: str,
    out_dir: Path,
    max_points: int = 20000,
    max_channels: int = 20,
):
    """
    Save an overview plot across the *entire* recording by decimating in time.
    This makes it feasible to view the whole run without massive files.

    - max_points caps the number of time points plotted per channel
    - max_channels limits displayed channels (first N EEG channels)
    """
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)

    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, misc=False)
    if max_channels is not None:
        picks = picks[: min(max_channels, len(picks))]

    decim = int(np.ceil(n_times / max_points)) if n_times > max_points else 1

    data = raw.get_data(picks=picks, start=0, stop=n_times)
    data = data[:, ::decim]
    times = (np.arange(data.shape[1]) * decim) / sfreq

    # Median-center each channel to remove DC offsets
    data_centered = data - np.median(data, axis=1, keepdims=True)

    # Robust scale from MAD of centered data
    med = np.median(data_centered)
    mad = np.median(np.abs(data_centered - med))
    scale = 6 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(data_centered) if np.std(data_centered) > 0 else 1e-6

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    offsets = np.arange(len(picks))[::-1]
    for i, p in enumerate(picks):
        ax.plot(times, data_centered[i] / scale + offsets[i], linewidth=0.5)

    ax.set_yticks(offsets)
    ax.set_yticklabels([raw.ch_names[p] for p in picks])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{fif_base}: Raw EEG overview (entire run, decim={decim}, median-centered, scaled)")
    ax.grid(True, alpha=0.2)

    out_path = out_dir / f"{fif_base}_raw_overview_decim{decim}_ch{len(picks)}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_psd_plot(raw: mne.io.BaseRaw, fif_base: str, out_dir: Path, fmin: float, fmax: float):
    """
    Save PSD plot to disk (uses MNE's PSD plotting, but non-interactive).
    PSD is computed across the recording by default unless raw is cropped.
    """
    psd = raw.compute_psd(fmin=fmin, fmax=fmax)
    fig = psd.plot(show=False, average=False)
    out_path = out_dir / f"{fif_base}_psd_{fmin:.1f}-{fmax:.1f}Hz.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_events_plot(events_df: pd.DataFrame, sfreq: float, fif_base: str, out_dir: Path):
    """
    Save events plot to disk (non-interactive).
    """
    if len(events_df) == 0:
        return
    events = events_df_to_mne_events(events_df, sample_col="sample0")
    fig = mne.viz.plot_events(events, sfreq=sfreq, first_samp=0, show=False)
    out_path = out_dir / f"{fif_base}_events.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -------------------------
# QC checks
# -------------------------
def qc_one_file(
    fif_path: Path,
    expected_event_codes: set | None,
    plots_dir: Path,
    plots_raw_snippet_dir: Path,
    plots_raw_overview_dir: Path,
    plots_psd_dir: Path,
    plots_events_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    events_csv, meta_json, base = find_sidecars(fif_path)
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")

    # Basic info
    nchan = int(raw.info["nchan"])
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    duration_sec = n_times / sfreq if sfreq else float("nan")
    ch_names = raw.ch_names
    ch_types = set(raw.get_channel_types())

    # Sidecars
    meta = load_meta_json(meta_json)
    events_df = load_events_csv(events_csv)

    # Checks: channel names unique
    unique_ok = (len(set(ch_names)) == len(ch_names))

    # Checks: expected channel types (eeg-only or eeg+eog both acceptable)
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, misc=False)
    eeg_count = len(eeg_picks)
    acceptable_types = (ch_types <= {"eeg", "eog"})  # only eeg and/or eog

    # Per-channel stats and correlation (decimated; uses QC_WINDOW_SEC or full recording)
    ch_stats = compute_channel_stats(raw, picks=eeg_picks, window_sec=QC_WINDOW_SEC, max_samples=MAX_SAMPLES_FOR_STATS)
    ch_corr = compute_mean_abs_correlation(raw, picks=eeg_picks, window_sec=QC_WINDOW_SEC, max_samples=MAX_SAMPLES_FOR_STATS)

    # Merge for unified channel-level output
    ch_df = ch_stats.merge(ch_corr, on=["ch_name", "qc_window_sec", "qc_samples_used", "decim"], how="left")
    ch_df.insert(0, "fif", str(fif_path))
    ch_df.insert(1, "fif_base", base)

    # --- Robust dead/near-dead channel detection (relative to run) ---
    EPS = 1e-20  # prevent divide-by-zero
    robust_std_ref = float(np.median(ch_df["std_v"].to_numpy()))
    robust_ptp_ref = float(np.median(ch_df["ptp_v"].to_numpy()))
    robust_mad_ref = float(np.median(ch_df["mad_v"].to_numpy()))

    # Ratios vs typical channel in this run
    ch_df["std_ratio_to_median"] = ch_df["std_v"] / (robust_std_ref + EPS)
    ch_df["ptp_ratio_to_median"] = ch_df["ptp_v"] / (robust_ptp_ref + EPS)
    ch_df["mad_ratio_to_median"] = ch_df["mad_v"] / (robust_mad_ref + EPS)

    ch_df["flag_flat_or_dead"] = (
        (ch_df["var_v2"] < NEAR_ZERO_VAR)
        | (ch_df["std_v"] < STD_MIN_V)
        | (ch_df["std_ratio_to_median"] < DEAD_STD_RATIO)
        | (ch_df["ptp_ratio_to_median"] < DEAD_PTP_RATIO)
        | (ch_df["mad_ratio_to_median"] < DEAD_MAD_RATIO)
    )

    # Channel flags
    ch_df["flag_std_high"] = ch_df["std_v"] > STD_MAX_V
    ch_df["flag_ptp_high"] = ch_df["ptp_v"] > PTP_MAX_V
    ch_df["flag_mad_high"] = ch_df["mad_v"] > MAD_MAX_V
    ch_df["flag_corr_low"] = ch_df["mean_abs_corr"] < MEAN_ABS_CORR_MIN
    ch_df["flag_corr_high"] = ch_df["mean_abs_corr"] > MEAN_ABS_CORR_MAX

    ch_df["any_flag"] = ch_df[
        [
            "flag_flat_or_dead",
            "flag_std_high",
            "flag_ptp_high",
            "flag_mad_high",
            "flag_corr_low",
            "flag_corr_high",
        ]
    ].any(axis=1)

    # Summarize flagged channels
    flagged_channels = ch_df.loc[ch_df["any_flag"], "ch_name"].tolist()

    # Events checks
    codes_present = set()
    in_bounds_ok = True
    if len(events_df) > 0:
        codes_present = set(map(int, events_df["code"].unique()))
        in_bounds_ok = bool(((events_df["sample0"] >= 0) & (events_df["sample0"] < n_times)).all())
    if expected_event_codes is not None:
        expected_codes_ok = (codes_present.issubset(expected_event_codes)) if codes_present else True
    else:
        expected_codes_ok = True  # no expected codes configured — skip check

    # Save plots (to per-file folder)
    per_file_plot_dir = plots_dir / "by_run" / base
    per_file_plot_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_RAW_SNIPPET_PLOT:
        save_raw_snippet_plot(raw, base, plots_raw_snippet_dir, duration_sec=RAW_SNIPPET_SEC, max_channels=None)
        save_raw_snippet_plot(raw, base, per_file_plot_dir, duration_sec=RAW_SNIPPET_SEC, max_channels=None)

    save_raw_overview_plot(raw, base, plots_raw_overview_dir, max_points=20000, max_channels=None)
    save_raw_overview_plot(raw, base, per_file_plot_dir, max_points=20000, max_channels=None)

    if SAVE_PSD_PLOT:
        save_psd_plot(raw, base, plots_psd_dir, fmin=PSD_FMIN, fmax=PSD_FMAX)
        save_psd_plot(raw, base, per_file_plot_dir, fmin=PSD_FMIN, fmax=PSD_FMAX)

    if SAVE_EVENTS_PLOT:
        save_events_plot(events_df, sfreq=sfreq, fif_base=base, out_dir=plots_events_dir)
        save_events_plot(events_df, sfreq=sfreq, fif_base=base, out_dir=per_file_plot_dir)


    report = {
        "fif": str(fif_path),
        "fif_base": base,
        "sfreq": sfreq,
        "nchan": nchan,
        "eeg_count": eeg_count,
        "n_times": n_times,
        "duration_sec": duration_sec,
        "unique_ch_names": unique_ok,
        "channel_types": sorted(list(ch_types)),
        "acceptable_ch_types": acceptable_types,
        "events_csv_exists": events_csv.exists(),
        "meta_json_exists": meta_json.exists(),
        "n_events": int(len(events_df)) if len(events_df) else 0,
        "codes_present": sorted(list(codes_present)) if codes_present else [],
        "expected_codes_subset": expected_codes_ok,
        "events_in_bounds": in_bounds_ok,
        "meta_input_units": meta.get("input_units"),
        "meta_stored_units": meta.get("stored_units"),
        # QC settings used
        "qc_window_sec": float(QC_WINDOW_SEC) if QC_WINDOW_SEC is not None else np.nan,
        "max_samples_for_stats": int(MAX_SAMPLES_FOR_STATS),
        # Channel flag summary
        "n_flagged_channels": int(len(flagged_channels)),
        "flagged_channels": ";".join(flagged_channels[:50]) + (";..." if len(flagged_channels) > 50 else ""),
        # Plot outputs
        "plots_dir": str(per_file_plot_dir),
    }

    return report, ch_df


def main():
    parser = argparse.ArgumentParser(
        description="Run QC inspection on converted raw FIF files."
    )
    parser.add_argument(
        "dataset_dir", type=Path,
        help="Path to dataset directory (must contain dataset_config.json "
             "and raw_fif/).",
    )
    parser.add_argument("--subject", type=str, default=None,
                        help="Only this subject (e.g. '01')")
    parser.add_argument("--session", type=str, default=None,
                        help="Only this session (e.g. '01')")
    parser.add_argument("--run", type=str, default=None,
                        help="Only this run (e.g. '01')")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    cfg_path = dataset_dir / "dataset_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No dataset_config.json found in {dataset_dir}"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)

    task = cfg.get("bids_task", "errp")

    # Derive expected event codes from config
    event_id = cfg.get("event_id", {})
    if cfg.get("input_format") == "matlab":
        expected_event_codes = set(event_id.values()) if event_id else None
    else:
        # eeglab: codes are auto-generated 1..N in conversion
        expected_event_codes = (
            set(range(1, len(event_id) + 1)) if event_id else None
        )

    raw_fif_root = dataset_dir / "raw_fif"
    qc_out_root = dataset_dir / "qc_outputs"
    qc_out_root.mkdir(parents=True, exist_ok=True)

    qc_summary_csv = qc_out_root / "qc_summary.csv"
    qc_channel_csv = qc_out_root / "qc_channel_stats.csv"
    plots_dir = qc_out_root / "qc_plots"
    plots_raw_snippet_dir = plots_dir / "raw_snippet"
    plots_raw_overview_dir = plots_dir / "raw_overview"
    plots_psd_dir = plots_dir / "psd"
    plots_events_dir = plots_dir / "events"

    for d in [plots_dir, plots_raw_snippet_dir, plots_raw_overview_dir,
              plots_psd_dir, plots_events_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fif_files = iter_fif_files(
        raw_fif_root, task,
        only_subject=args.subject,
        only_session=args.session,
        only_run=args.run,
    )
    if not fif_files:
        raise FileNotFoundError(f"No FIF files found under {raw_fif_root}")

    print(f"Dataset: {cfg.get('dataset_name', dataset_dir.name)}")
    print(f"Found {len(fif_files)} FIF files under: {raw_fif_root}\n")
    print(f"QC outputs will be written to: {qc_out_root.resolve()}\n")

    reports = []
    channel_rows = []

    for p in fif_files:
        rep, ch_df = qc_one_file(
            p,
            expected_event_codes=expected_event_codes,
            plots_dir=plots_dir,
            plots_raw_snippet_dir=plots_raw_snippet_dir,
            plots_raw_overview_dir=plots_raw_overview_dir,
            plots_psd_dir=plots_psd_dir,
            plots_events_dir=plots_events_dir,
        )
        reports.append(rep)
        channel_rows.append(ch_df)

        print(
            f"{Path(rep['fif']).name} | "
            f"{rep['eeg_count']} EEG ch | {rep['sfreq']} Hz | "
            f"{rep['duration_sec']:.1f}s | "
            f"{rep['n_events']} events | "
            f"flagged_ch={rep['n_flagged_channels']}"
        )

        problems = []
        if not rep["unique_ch_names"]:
            problems.append("duplicate channel names")
        if not rep["acceptable_ch_types"]:
            problems.append(
                f"unexpected channel types {rep['channel_types']}"
            )
        if not rep["events_csv_exists"]:
            problems.append("missing events_csv")
        if not rep["meta_json_exists"]:
            problems.append("missing meta_json")
        if not rep["events_in_bounds"]:
            problems.append("events out of bounds")
        if not rep["expected_codes_subset"]:
            problems.append(
                f"unexpected event codes {rep['codes_present']}"
            )
        if rep["n_flagged_channels"] > 0:
            problems.append(
                f"{rep['n_flagged_channels']} flagged EEG channels"
            )

        if problems:
            print("  [!] QC flags:", "; ".join(problems))

    # Write summary CSVs
    pd.DataFrame(reports).to_csv(qc_summary_csv, index=False)
    print(f"\nWrote QC summary: {qc_summary_csv.resolve()}")

    all_ch = (pd.concat(channel_rows, ignore_index=True)
              if channel_rows else pd.DataFrame())
    all_ch.to_csv(qc_channel_csv, index=False)
    print(f"Wrote per-channel QC: {qc_channel_csv.resolve()}")


if __name__ == "__main__":
    main()