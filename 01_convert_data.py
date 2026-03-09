"""
Dataset -> MNE Raw -> FIF + BIDS EEG export (BrainVision)

Config-driven conversion that supports multiple input formats:
  - matlab  : BNCI-style .mat files (multi-run per file)
  - eeglab  : EEGLAB .set/.fdt files (single recording per file)

Each dataset directory must contain:
  - dataset_config.json  (format, event IDs, channel info, etc.)
  - original_data/       (raw source files)

Usage:
  python 01_convert_data.py datasets/bnci_horizon_2020_ErrP
  python 01_convert_data.py datasets/hri_cursor
  python 01_convert_data.py datasets/bnci_horizon_2020_ErrP --export-mode concatenate

Requirements:
  pip install mne mne-bids scipy numpy pandas pybv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import mne


# ============================================================================
# Config loading
# ============================================================================

def load_dataset_config(dataset_dir: Path) -> dict:
    """Load and validate dataset_config.json from a dataset directory."""
    cfg_path = dataset_dir / "dataset_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No dataset_config.json found in {dataset_dir}. "
            "Each dataset directory must contain a config file."
        )
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Validate required fields
    required = [
        "input_format", "file_glob", "filename_regex",
        "channel_type", "bids_task", "export_mode", "event_id",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"dataset_config.json missing required fields: {missing}")

    if cfg["input_format"] not in ("matlab", "eeglab"):
        raise ValueError(f"Unsupported input_format: {cfg['input_format']!r}")
    if cfg["export_mode"] not in ("per_run", "concatenate", "single"):
        raise ValueError(f"Unsupported export_mode: {cfg['export_mode']!r}")

    return cfg


def parse_filename(filename: str, regex: str) -> dict[str, str]:
    """Extract subject (and optionally session) from filename using regex."""
    m = re.match(regex, filename, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Filename {filename!r} does not match regex {regex!r}"
        )
    groups = m.groupdict()
    # Pad subject to 2 digits
    groups["sub"] = f"{int(groups['sub']):02d}"
    # Pad session if present, else default to "01"
    if "ses" in groups:
        groups["ses"] = f"{int(groups['ses']):02d}"
    else:
        groups["ses"] = "01"
    return groups


# ============================================================================
# MATLAB (.mat) loading — BNCI-style multi-run files
# ============================================================================

def _unwrap(v):
    if isinstance(v, np.ndarray):
        v = np.array(v).squeeze()
        if v.dtype == object and v.size == 1:
            v = v.item()
        if isinstance(v, np.ndarray) and v.size == 1:
            v = v.item()
    return v


def _as_1d_array(v):
    v = _unwrap(v)
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return np.array(v).ravel()
    if isinstance(v, np.ndarray):
        return np.array(v).ravel()
    return np.array([v]).ravel()


def _matstruct_fields(ms):
    if hasattr(ms, "__dict__"):
        return [k for k in ms.__dict__.keys() if not k.startswith("_")]
    return [k for k in dir(ms) if not k.startswith("_")]


def _get_field_case_insensitive(ms, name: str):
    fields = _matstruct_fields(ms)
    lower_map = {f.lower(): f for f in fields}
    key = lower_map.get(name.lower())
    return getattr(ms, key) if key else None


def _to_str_list(v):
    arr = _as_1d_array(v)
    if arr is None:
        return None
    return [str(_unwrap(x)) for x in arr]


def _extract_eeg_mat(run_item) -> np.ndarray:
    eeg = np.array(run_item.eeg)
    if eeg.ndim != 2:
        raise ValueError(f"Expected 2D EEG, got shape {eeg.shape}")
    return eeg  # (samples, channels)


def _extract_header_mat(run_item) -> dict:
    hdr = run_item.header
    meta = {}
    if hasattr(hdr, "Subject"):
        meta["subject"] = str(_unwrap(hdr.Subject))
    if hasattr(hdr, "Session"):
        meta["session"] = str(_unwrap(hdr.Session))
    if hasattr(hdr, "SampleRate"):
        meta["sfreq"] = float(_unwrap(hdr.SampleRate))
    if hasattr(hdr, "Label"):
        labels = _to_str_list(hdr.Label)
        if labels:
            meta["ch_names"] = labels
    meta["header_fields_present"] = _matstruct_fields(hdr)
    return meta


def _extract_events_mat(run_item) -> pd.DataFrame:
    """Return DataFrame with sample_matlab (1-based), sample0 (0-based), code (int)."""
    hdr = run_item.header
    event_ms = getattr(hdr, "EVENT", None)
    if event_ms is None:
        return pd.DataFrame(columns=["sample_matlab", "sample0", "code"])

    pos = _get_field_case_insensitive(event_ms, "POS")
    typ = _get_field_case_insensitive(event_ms, "TYP")

    if pos is None or typ is None:
        raise ValueError(
            f"EVENT missing POS or TYP. Fields: {_matstruct_fields(event_ms)}"
        )

    pos = _as_1d_array(pos).astype(float)
    typ = _as_1d_array(typ)

    if len(pos) != len(typ):
        raise ValueError(f"POS/TYP length mismatch: {len(pos)} vs {len(typ)}")

    sample_matlab = np.round(pos).astype(int)
    sample0 = sample_matlab - 1
    codes = [int(round(float(_unwrap(x)))) for x in typ]

    return pd.DataFrame({
        "sample_matlab": sample_matlab,
        "sample0": sample0,
        "code": codes,
    })


def load_mat_runs(mat_path: Path):
    from scipy.io import loadmat
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "run" not in mat:
        raise KeyError(
            f"No 'run' in MAT. Keys: "
            f"{[k for k in mat.keys() if not k.startswith('__')]}"
        )
    run = mat["run"]
    if not hasattr(run, "shape"):
        run = np.array([run], dtype=object)
    return np.array(run).reshape(-1)


def _parse_matlab_common(mat_path: Path, cfg: dict):
    """Shared setup for MATLAB loading: parse runs, sfreq, ch_names."""
    runs = load_mat_runs(mat_path)
    n_runs = int(runs.shape[0])
    assume_uv = cfg.get("assume_microvolts", True)
    ch_type = cfg.get("channel_type", "eeg")
    montage_name = cfg.get("montage", "standard_1005")
    code_to_desc = {v: k for k, v in cfg["event_id"].items()}

    meta0 = _extract_header_mat(runs[0])
    sfreq = meta0.get("sfreq")
    if sfreq is None:
        raise ValueError("Could not read SampleRate (sfreq) from header.")

    n_ch = _extract_eeg_mat(runs[0]).shape[1]

    ch_names = meta0.get("ch_names")
    if ch_names:
        if len(ch_names) == n_ch + 1:
            ch_names = ch_names[:n_ch]
        if len(ch_names) != n_ch:
            ch_names = None
    if not ch_names:
        ch_names = [f"ch_{i}" for i in range(n_ch)]

    return runs, n_runs, sfreq, ch_names, n_ch, assume_uv, ch_type, montage_name, code_to_desc


def _get_run_eeg(run_item, i: int, n_ch: int, assume_uv: bool) -> np.ndarray:
    """Extract, validate, and unit-convert EEG from one MATLAB run."""
    eeg = _extract_eeg_mat(run_item)
    if eeg.shape[1] != n_ch:
        raise ValueError(f"Run {i} channel count mismatch: {eeg.shape[1]} vs {n_ch}")
    if assume_uv:
        eeg = eeg * 1e-6
    return eeg


def _make_raw(
    data_T: np.ndarray, ch_names: list[str], sfreq: float,
    ch_type: str, montage_name: str,
) -> mne.io.RawArray:
    """Create MNE RawArray with montage from (channels, samples) data."""
    info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types=[ch_type] * len(ch_names),
    )
    raw = mne.io.RawArray(data_T, info, verbose=False)
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="warn")
    return raw


def _events_df_to_array(events_df: pd.DataFrame, sample_col: str = "sample0") -> np.ndarray:
    """Convert events DataFrame to MNE (N, 3) events array."""
    if len(events_df) == 0:
        return np.zeros((0, 3), dtype=int)
    return np.column_stack([
        events_df[sample_col].astype(int).to_numpy(),
        np.zeros(len(events_df), dtype=int),
        events_df["code"].astype(int).to_numpy(),
    ])


def load_matlab_per_run(
    mat_path: Path, cfg: dict
) -> list[tuple[mne.io.RawArray, np.ndarray, pd.DataFrame, dict, int]]:
    """Load a BNCI-style .mat and yield (raw, events, events_df, meta, run_idx1) per run."""
    runs, n_runs, sfreq, ch_names, n_ch, assume_uv, ch_type, montage_name, code_to_desc = \
        _parse_matlab_common(mat_path, cfg)

    results = []
    for i in range(n_runs):
        eeg = _get_run_eeg(runs[i], i, n_ch, assume_uv)
        raw = _make_raw(eeg.T, ch_names, sfreq, ch_type, montage_name)

        events_df = _extract_events_mat(runs[i])
        events_df["mat_run_index0"] = i
        events_df["mat_run_index1"] = i + 1
        events = _events_df_to_array(events_df)

        if len(events_df) > 0:
            onsets = events_df["sample0"].astype(float).to_numpy() / float(sfreq)
            descriptions = np.array([
                code_to_desc.get(int(c), str(int(c)))
                for c in events_df["code"].to_numpy()
            ])
            raw.set_annotations(mne.Annotations(
                onset=onsets,
                duration=np.zeros(len(events_df), dtype=float),
                description=descriptions,
            ))

        meta = {
            "source_file": str(mat_path), "input_format": "matlab",
            "export_mode": "per_run", "n_runs_in_mat": n_runs,
            "mat_run_index0": int(i), "mat_run_index1": int(i + 1),
            "n_samples": int(eeg.shape[0]), "n_channels": int(n_ch),
            "sfreq": float(sfreq), "ch_names": ch_names,
            "input_units": "uV" if assume_uv else "V", "stored_units": "V",
        }
        results.append((raw, events, events_df, meta, i + 1))

    return results


def load_matlab_concatenated(
    mat_path: Path, cfg: dict
) -> tuple[mne.io.RawArray, np.ndarray, pd.DataFrame, dict]:
    """Load a BNCI-style .mat and concatenate all runs into one recording."""
    runs, n_runs, sfreq, ch_names, n_ch, assume_uv, ch_type, montage_name, code_to_desc = \
        _parse_matlab_common(mat_path, cfg)

    eeg_runs = []
    samples_per_run = []
    for i in range(n_runs):
        eeg = _get_run_eeg(runs[i], i, n_ch, assume_uv)
        eeg_runs.append(eeg)
        samples_per_run.append(int(eeg.shape[0]))

    raw = _make_raw(np.concatenate(eeg_runs, axis=0).T, ch_names, sfreq, ch_type, montage_name)

    run_offsets = np.cumsum([0] + samples_per_run[:-1])
    all_events = []
    for i in range(n_runs):
        ev = _extract_events_mat(runs[i])
        if len(ev) == 0:
            continue
        ev["mat_run_index0"] = i
        ev["mat_run_index1"] = i + 1
        ev["sample_global0"] = run_offsets[i] + ev["sample0"]
        all_events.append(ev)

    events_df = (
        pd.concat(all_events, ignore_index=True) if all_events
        else pd.DataFrame(columns=[
            "sample_matlab", "sample0", "code",
            "mat_run_index0", "mat_run_index1", "sample_global0",
        ])
    )
    events = _events_df_to_array(events_df, sample_col="sample_global0")

    if len(events_df) > 0:
        onsets = events_df["sample_global0"].astype(float).to_numpy() / float(sfreq)
        descriptions = np.array([
            code_to_desc.get(int(c), str(int(c)))
            for c in events_df["code"].to_numpy()
        ])
        raw.set_annotations(mne.Annotations(
            onset=onsets,
            duration=np.zeros(len(events_df), dtype=float),
            description=descriptions,
        ))

    meta = {
        "source_file": str(mat_path), "input_format": "matlab",
        "export_mode": "concatenate", "n_runs_in_mat": int(n_runs),
        "samples_per_run": samples_per_run,
        "n_samples_total": int(sum(samples_per_run)),
        "n_channels": int(n_ch), "sfreq": float(sfreq), "ch_names": ch_names,
        "input_units": "uV" if assume_uv else "V", "stored_units": "V",
    }

    return raw, events, events_df, meta


# ============================================================================
# EEGLAB (.set/.fdt) loading
# ============================================================================

def load_eeglab_single(
    set_path: Path, cfg: dict
) -> tuple[mne.io.BaseRaw, np.ndarray, pd.DataFrame, dict]:
    """Load an EEGLAB .set file and extract events from annotations."""
    eog_channels = cfg.get("eog_channels", [])
    montage_name = cfg.get("montage", "standard_1020")

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

    # Set EOG channel types
    if eog_channels:
        present_eog = [ch for ch in eog_channels if ch in raw.ch_names]
        if present_eog:
            raw.set_channel_types({ch: "eog" for ch in present_eog})

    # Set montage (only for EEG channels)
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="warn")

    # Build event_id: in EEGLAB configs, event_id maps name -> annotation string
    # We need to convert annotation strings to integer codes for MNE events
    event_id_cfg = cfg["event_id"]

    # For EEGLAB: event_id values are the annotation description strings
    # We create a mapping: description_string -> integer_code
    desc_to_code = {}
    desc_to_name = {}
    for i, (name, desc_str) in enumerate(event_id_cfg.items(), start=1):
        desc_to_code[desc_str] = i
        desc_to_name[desc_str] = name

    # Extract events from annotations and remap descriptions to names
    annot = raw.annotations
    rows = []
    new_onsets = []
    new_durations = []
    new_descriptions = []

    for a in annot:
        desc = a["description"]
        if desc in desc_to_code:
            sample0 = int(round(a["onset"] * raw.info["sfreq"]))
            name = desc_to_name[desc]
            rows.append({
                "sample0": sample0,
                "code": desc_to_code[desc],
                "description_original": desc,
                "name": name,
                "onset_sec": float(a["onset"]),
            })
            new_onsets.append(a["onset"])
            new_durations.append(a["duration"])
            new_descriptions.append(name)

    events_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["sample0", "code", "description_original", "name", "onset_sec"]
    )

    if len(events_df) > 0:
        events = np.column_stack([
            events_df["sample0"].astype(int).to_numpy(),
            np.zeros(len(events_df), dtype=int),
            events_df["code"].astype(int).to_numpy(),
        ])
    else:
        events = np.zeros((0, 3), dtype=int)

    # Replace raw annotations with our mapped names (only known events)
    # so the BIDS writer sees consistent event descriptions
    raw.set_annotations(mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descriptions,
    ))

    # Build the name-based event_id for BIDS (name -> integer code)
    event_id_bids = {name: desc_to_code[desc_str]
                     for name, desc_str in event_id_cfg.items()}

    meta = {
        "source_file": str(set_path),
        "input_format": "eeglab",
        "export_mode": "single",
        "n_samples": int(raw.n_times),
        "n_channels": int(raw.info["nchan"]),
        "sfreq": float(raw.info["sfreq"]),
        "ch_names": raw.ch_names,
        "eog_channels": eog_channels,
        "stored_units": "V",
        "event_id_bids": event_id_bids,
    }

    return raw, events, events_df, meta


# ============================================================================
# Save helpers
# ============================================================================

def save_fif(
    raw: mne.io.BaseRaw, out_root: Path,
    subject: str, session: str, task: str, run: str,
) -> Path:
    out_dir = out_root / f"sub-{subject}" / f"ses-{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fif_path = out_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_raw.fif"
    raw.save(fif_path, overwrite=True)
    return fif_path


def save_aux_files(
    out_root: Path, subject: str, session: str, task: str, run: str,
    events_df: pd.DataFrame, meta: dict,
) -> tuple[Path, Path]:
    aux_dir = out_root / f"sub-{subject}" / f"ses-{session}"
    aux_dir.mkdir(parents=True, exist_ok=True)

    events_csv = aux_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.csv"
    events_df.to_csv(events_csv, index=False)

    meta_json = aux_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_meta.json"
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    return events_csv, meta_json


def save_bids(
    raw: mne.io.BaseRaw, events: np.ndarray, event_id: dict,
    out_root: Path, subject: str, session: str, task: str, run: str,
) -> "BIDSPath":
    from mne_bids import BIDSPath, write_raw_bids

    bids_path = BIDSPath(
        subject=subject, session=session, task=task, run=run,
        datatype="eeg", root=out_root,
    )

    write_raw_bids(
        raw,
        bids_path=bids_path,
        events=events if len(events) else None,
        event_id=event_id,
        overwrite=True,
        verbose=False,
        allow_preload=True,
        format="BrainVision",
    )
    return bids_path


# ============================================================================
# Processing dispatch
# ============================================================================

def process_matlab_file(
    mat_path: Path, cfg: dict, export_mode: str,
    out_fif: Path, out_bids: Path,
):
    """Process a single .mat file (BNCI-style)."""
    groups = parse_filename(mat_path.name, cfg["filename_regex"])
    subject, session = groups["sub"], groups["ses"]
    task = cfg["bids_task"]
    event_id = cfg["event_id"]

    print(f"\n=== Processing {mat_path.name} -> "
          f"sub-{subject} ses-{session} (mode={export_mode}) ===")

    if export_mode == "per_run":
        for raw, events, events_df, meta, run_idx1 in \
                load_matlab_per_run(mat_path, cfg):
            bids_run = f"{run_idx1:02d}"
            print(f"--- MAT run {run_idx1} -> BIDS run {bids_run} ---")

            fif_path = save_fif(raw, out_fif, subject, session, task, bids_run)
            events_csv, meta_json = save_aux_files(
                out_fif, subject, session, task, bids_run, events_df, meta,
            )
            bids_path = save_bids(
                raw, events, event_id, out_bids,
                subject, session, task, bids_run,
            )
            _print_summary(fif_path, events_csv, meta_json, bids_path,
                           events, raw)

    else:  # concatenate
        raw, events, events_df, meta = load_matlab_concatenated(mat_path, cfg)
        bids_run = "01"

        fif_path = save_fif(raw, out_fif, subject, session, task, bids_run)
        events_csv, meta_json = save_aux_files(
            out_fif, subject, session, task, bids_run, events_df, meta,
        )
        bids_path = save_bids(
            raw, events, event_id, out_bids,
            subject, session, task, bids_run,
        )
        _print_summary(fif_path, events_csv, meta_json, bids_path,
                       events, raw)


def process_eeglab_file(
    set_path: Path, cfg: dict,
    out_fif: Path, out_bids: Path,
):
    """Process a single EEGLAB .set file."""
    groups = parse_filename(set_path.name, cfg["filename_regex"])
    subject, session = groups["sub"], groups["ses"]
    task = cfg["bids_task"]

    print(f"\n=== Processing {set_path.name} -> "
          f"sub-{subject} ses-{session} ===")

    raw, events, events_df, meta = load_eeglab_single(set_path, cfg)
    event_id_bids = meta.get("event_id_bids", {})
    bids_run = "01"

    fif_path = save_fif(raw, out_fif, subject, session, task, bids_run)
    events_csv, meta_json = save_aux_files(
        out_fif, subject, session, task, bids_run, events_df, meta,
    )
    bids_path = save_bids(
        raw, events, event_id_bids, out_bids,
        subject, session, task, bids_run,
    )
    _print_summary(fif_path, events_csv, meta_json, bids_path, events, raw)


def _print_summary(fif_path, events_csv, meta_json, bids_path, events, raw):
    print(f"  FIF:  {fif_path}")
    print(f"  CSV:  {events_csv}")
    print(f"  META: {meta_json}")
    print(f"  BIDS: {bids_path.directory}")
    print(f"  Sanity: {events.shape[0]} events, "
          f"{raw.n_times} samples, {raw.info['nchan']} ch")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert raw EEG data to MNE FIF + BIDS format."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to dataset directory (must contain dataset_config.json "
             "and original_data/).",
    )
    parser.add_argument(
        "--export-mode",
        type=str,
        default=None,
        help="Override export_mode from config "
             "(per_run | concatenate | single).",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    cfg = load_dataset_config(dataset_dir)

    export_mode = args.export_mode or cfg["export_mode"]
    input_format = cfg["input_format"]
    file_glob = cfg["file_glob"]

    input_dir = dataset_dir / "original_data"
    out_fif = dataset_dir / "raw_fif"
    out_bids = dataset_dir / "bids"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    source_files = sorted(input_dir.glob(file_glob))
    if not source_files:
        raise FileNotFoundError(
            f"No files matching {file_glob!r} in {input_dir}"
        )

    out_fif.mkdir(parents=True, exist_ok=True)
    (out_fif / ".gitkeep").touch()
    out_bids.mkdir(parents=True, exist_ok=True)
    (out_bids / ".gitkeep").touch()

    print(f"Dataset:      {cfg.get('dataset_name', dataset_dir.name)}")
    print(f"Input format: {input_format}")
    print(f"Export mode:  {export_mode}")
    print(f"Source files: {len(source_files)}")
    print(f"Input dir:    {input_dir}")
    print(f"FIF output:   {out_fif}")
    print(f"BIDS output:  {out_bids}")

    for source_path in source_files:
        if input_format == "matlab":
            process_matlab_file(
                source_path, cfg, export_mode, out_fif, out_bids,
            )
        elif input_format == "eeglab":
            process_eeglab_file(source_path, cfg, out_fif, out_bids)


if __name__ == "__main__":
    main()
