"""Chronological CHB-MIT seizure-prediction dataset for NeuroLens.

Builds strictly non-overlapping, chronologically ordered 5-second windows
from the CHB-MIT Scalp EEG database, labeled for seizure PREDICTION (not
detection):

    - Preictal window:  fully inside [onset - 60 min, onset - 5 min).
    - Interictal window: >= 4 hours from every seizure's onset *and* offset.
    - Everything else (ictal activity, the 5-minute Seizure Prediction
      Horizon (SPH) immediately before onset, and the buffer between the
      preictal window and the 4-hour interictal boundary) is excluded from
      both training and evaluation, so no window can leak seizure-detection
      information into a seizure-prediction task.

Seizure annotations are parsed from each patient's ``chbXX-summary.txt``,
the canonical, complete, human-readable annotation source PhysioNet ships
for every CHB-MIT recording (the undocumented per-file binary
``.seizures`` sidecars are a redundant, unofficial format and are not
used here).
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

mne.set_log_level("ERROR")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

FS = 256                                   # Hz, target sampling rate
WINDOW_SEC = 5                             # non-overlapping epoch length
N_SAMPLES = FS * WINDOW_SEC                # 1280 samples/channel/epoch

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 45.0
BANDPASS_ORDER = 4                         # 4th-order Butterworth
NOTCH_FREQ_HZ = 60.0

PREICTAL_START_MIN = 60.0                  # preictal window opens 60 min before onset
SPH_MIN = 5.0                              # Seizure Prediction Horizon, excluded entirely
INTERICTAL_BUFFER_HOURS = 4.0              # required isolation from any seizure

LABEL_INTERICTAL = 0
LABEL_PREICTAL = 1

# Standard 18-channel double-banana bipolar montage.
BIPOLAR_MONTAGE: List[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
N_CHANNELS = len(BIPOLAR_MONTAGE)


# --------------------------------------------------------------------------
# chbXX-summary.txt parsing
# --------------------------------------------------------------------------

_FILE_NAME_RE = re.compile(r"File Name:\s*(\S+)")
_FILE_START_RE = re.compile(r"File Start Time:\s*([\d:]+)")
_FILE_END_RE = re.compile(r"File End Time:\s*([\d:]+)")
_SEIZURE_START_RE = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", re.IGNORECASE)
_SEIZURE_END_RE = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", re.IGNORECASE)


@dataclass
class SummaryFileEntry:
    filename: str
    start_time_str: Optional[str]
    end_time_str: Optional[str]
    seizures_sec: List[Tuple[float, float]]  # (onset, offset) seconds relative to file start


def parse_summary_file(summary_path: str) -> List[SummaryFileEntry]:
    """Parse a chbXX-summary.txt into one entry per recorded EDF file.

    Robust to both the single-seizure ("Seizure Start Time") and the
    multi-seizure ("Seizure 1 Start Time", "Seizure 2 Start Time", ...)
    phrasing used across different CHB-MIT patient summaries.
    """
    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Split into per-file blocks at each "File Name:" occurrence.
    starts = [m.start() for m in _FILE_NAME_RE.finditer(text)]
    entries: List[SummaryFileEntry] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end]

        filename = _FILE_NAME_RE.search(block).group(1)
        start_m = _FILE_START_RE.search(block)
        end_m = _FILE_END_RE.search(block)
        onsets = [float(x) for x in _SEIZURE_START_RE.findall(block)]
        offsets = [float(x) for x in _SEIZURE_END_RE.findall(block)]
        if len(onsets) != len(offsets):
            raise ValueError(
                f"Mismatched seizure start/end count for {filename} in {summary_path}"
            )
        seizures = list(zip(onsets, offsets))

        entries.append(
            SummaryFileEntry(
                filename=filename,
                start_time_str=start_m.group(1) if start_m else None,
                end_time_str=end_m.group(1) if end_m else None,
                seizures_sec=seizures,
            )
        )
    return entries


def _time_of_day_seconds(hms: Optional[str]) -> Optional[float]:
    """Parse HH:MM:SS into seconds-of-day; None if missing/unparseable.

    CHB-MIT headers occasionally record hours >= 24 (a known dataset quirk);
    these are folded modulo 24h rather than treated as a parse failure.
    """
    if not hms:
        return None
    try:
        h, m, s = hms.strip().split(":")
        return (int(h) % 24) * 3600.0 + int(m) * 60.0 + int(s)
    except (ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------
# Patient-level chronological timeline
# --------------------------------------------------------------------------


@dataclass
class FileRecord:
    filename: str
    filepath: str
    abs_start_sec: float
    abs_end_sec: float
    duration_sec: float
    seizures_abs: List[Tuple[float, float]]  # (onset, offset) in patient-absolute seconds


@dataclass
class WindowRecord:
    file_idx: int
    start_sample: int          # sample offset into the file's 256 Hz, N_SAMPLES-aligned timeline
    abs_start_sec: float
    label: int                 # LABEL_INTERICTAL or LABEL_PREICTAL
    seizure_idx: int           # index into the patient's chronological seizure list, or -1


def _edf_duration_sec(filepath: str) -> float:
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    return raw.n_times / raw.info["sfreq"]


def build_patient_timeline(patient_id: str, data_dir: str) -> List[FileRecord]:
    """Reconstruct the patient's chronological recording timeline.

    File order follows the summary.txt block order (the curated,
    chronological order used throughout CHB-MIT). Absolute start times are
    reconstructed from each file's "File Start Time" header with
    day-wraparound handling; whenever that reconstruction is missing or
    implausible (a known issue with a subset of CHB-MIT headers), the file
    is conservatively placed immediately after the previous file ends --
    the standard back-to-back assumption used throughout the seizure-
    prediction literature for this dataset.
    """
    summary_path = os.path.join(data_dir, patient_id, f"{patient_id}-summary.txt")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    entries = parse_summary_file(summary_path)

    records: List[FileRecord] = []
    prev_end_abs = 0.0
    prev_tod: Optional[float] = None
    day_offset = 0.0
    max_plausible_gap_sec = 6.0 * 3600.0

    for entry in entries:
        filepath = os.path.join(data_dir, patient_id, entry.filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"EDF file listed in summary not found: {filepath}")

        duration_sec = _edf_duration_sec(filepath)
        tod = _time_of_day_seconds(entry.start_time_str)

        abs_start: float
        if tod is None or prev_tod is None:
            abs_start = prev_end_abs
        else:
            if tod < prev_tod:
                day_offset += 86400.0
            candidate = day_offset + tod
            gap = candidate - prev_end_abs
            if gap < 0.0 or gap > max_plausible_gap_sec:
                abs_start = prev_end_abs
            else:
                abs_start = candidate

        abs_end = abs_start + duration_sec
        seizures_abs = [(abs_start + on, abs_start + off) for on, off in entry.seizures_sec]

        records.append(
            FileRecord(
                filename=entry.filename,
                filepath=filepath,
                abs_start_sec=abs_start,
                abs_end_sec=abs_end,
                duration_sec=duration_sec,
                seizures_abs=seizures_abs,
            )
        )
        prev_end_abs = abs_end
        if tod is not None:
            prev_tod = tod

    return records


def _all_seizures_chronological(records: List[FileRecord]) -> List[Tuple[float, float]]:
    seizures = [s for rec in records for s in rec.seizures_abs]
    seizures.sort(key=lambda s: s[0])
    return seizures


def _label_window(
    win_start_abs: float,
    win_end_abs: float,
    all_seizures: List[Tuple[float, float]],
) -> Optional[Tuple[int, int]]:
    """Label one window, or None if it must be excluded.

    Returns (label, seizure_idx). seizure_idx is the chronological index of
    the seizure a PREICTAL window belongs to, or -1 for INTERICTAL.
    """
    sph_sec = SPH_MIN * 60.0
    preictal_span_sec = PREICTAL_START_MIN * 60.0
    interictal_buffer_sec = INTERICTAL_BUFFER_HOURS * 3600.0

    # Exclude anything overlapping the SPH or the ictal period of any seizure.
    for onset, offset in all_seizures:
        sph_start = onset - sph_sec
        if win_end_abs > sph_start and win_start_abs < offset:
            return None

    # Preictal: fully inside [onset - 60min, onset - 5min) of exactly one seizure.
    for idx, (onset, _offset) in enumerate(all_seizures):
        pre_start = onset - preictal_span_sec
        pre_end = onset - sph_sec
        if win_start_abs >= pre_start and win_end_abs <= pre_end:
            return LABEL_PREICTAL, idx

    # Interictal: outside the +/-4h zone of every seizure's onset/offset span.
    for onset, offset in all_seizures:
        danger_start = onset - interictal_buffer_sec
        danger_end = offset + interictal_buffer_sec
        if win_end_abs > danger_start and win_start_abs < danger_end:
            return None  # inside some seizure's exclusion buffer -> drop

    return LABEL_INTERICTAL, -1


def build_patient_windows(
    patient_id: str, data_dir: str
) -> Tuple[List[FileRecord], List[WindowRecord], List[Tuple[float, float]]]:
    """Build the full chronological, labeled window index for one patient.

    Windows never cross a file boundary (each file's own bandpass/notch
    filtering must stay contiguous), and every window is independently
    checked against every seizure in the patient's record for SPH/ictal
    exclusion, preictal membership, and the interictal isolation buffer.
    """
    records = build_patient_timeline(patient_id, data_dir)
    all_seizures = _all_seizures_chronological(records)

    windows: List[WindowRecord] = []
    for file_idx, rec in enumerate(records):
        n_windows = int(rec.duration_sec // WINDOW_SEC)
        for k in range(n_windows):
            win_start_abs = rec.abs_start_sec + k * WINDOW_SEC
            win_end_abs = win_start_abs + WINDOW_SEC
            result = _label_window(win_start_abs, win_end_abs, all_seizures)
            if result is None:
                continue
            label, seizure_idx = result
            windows.append(
                WindowRecord(
                    file_idx=file_idx,
                    start_sample=k * N_SAMPLES,
                    abs_start_sec=win_start_abs,
                    label=label,
                    seizure_idx=seizure_idx,
                )
            )

    return records, windows, all_seizures


def group_contiguous_runs(windows: List[WindowRecord]) -> List[List[WindowRecord]]:
    """Split a window list into maximal runs that are truly temporally
    contiguous (same file, exact WINDOW_SEC stride).

    The exclusion rules in _label_window (SPH, interictal buffer, ictal
    period) mean consecutive entries in an arbitrary filtered window list
    -- e.g. "all interictal windows in this fold's training set" -- are NOT
    guaranteed adjacent in time. Anything that needs genuine temporal
    continuity (embedding a real EEG stretch, building FAISS trajectory
    segments) must regroup into contiguous runs first, which is what this
    does.
    """
    ordered = sorted(windows, key=lambda w: (w.file_idx, w.start_sample))
    runs: List[List[WindowRecord]] = []
    current: List[WindowRecord] = []
    for w in ordered:
        if (
            current
            and current[-1].file_idx == w.file_idx
            and abs((w.abs_start_sec - current[-1].abs_start_sec) - WINDOW_SEC) < 1e-6
        ):
            current.append(w)
        else:
            if current:
                runs.append(current)
            current = [w]
    if current:
        runs.append(current)
    return runs


# --------------------------------------------------------------------------
# EDF loading and preprocessing
# --------------------------------------------------------------------------


def _resolve_channel_name(available: List[str], canonical: str) -> str:
    lookup: Dict[str, str] = {ch.strip().upper(): ch for ch in available}
    key = canonical.strip().upper()
    if key in lookup:
        return lookup[key]
    for suffix in ("-0", "-1", "-2"):  # mne's disambiguation suffix for duplicate labels
        if key + suffix in lookup:
            return lookup[key + suffix]
    raise ValueError(f"Required channel '{canonical}' not found among {available}")


def load_bipolar_preprocessed(filepath: str) -> np.ndarray:
    """Load one EDF, enforce the 18-channel bipolar montage, and preprocess.

    Pipeline: pick + reorder to BIPOLAR_MONTAGE -> resample to FS Hz ->
    zero-phase 4th-order Butterworth bandpass (0.5-45 Hz) -> 60 Hz notch.

    Returns a (N_CHANNELS, n_samples) float32 array in microvolts.
    """
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw.rename_channels({ch: ch.strip() for ch in raw.ch_names})

    resolved = [_resolve_channel_name(raw.ch_names, name) for name in BIPOLAR_MONTAGE]
    raw.pick(resolved)
    raw.reorder_channels(resolved)

    if abs(raw.info["sfreq"] - FS) > 1e-6:
        raw.resample(FS, verbose=False)

    raw.filter(
        l_freq=BANDPASS_LOW_HZ,
        h_freq=BANDPASS_HIGH_HZ,
        method="iir",
        iir_params=dict(order=BANDPASS_ORDER, ftype="butter"),
        verbose=False,
    )
    raw.notch_filter(freqs=NOTCH_FREQ_HZ, verbose=False)

    data_uv = raw.get_data() * 1e6  # volts -> microvolts
    return data_uv.astype(np.float32)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class CHBMITChronologicalDataset(Dataset):
    """Windowed CHB-MIT seizure-prediction dataset over a fixed window list.

    Wraps a pre-built, chronologically labeled ``List[WindowRecord]`` (see
    :func:`build_patient_windows` / :func:`get_loso_splits`) and lazily
    loads + preprocesses each referenced EDF file on first access, caching
    a small number of most-recently-used files to keep memory bounded
    while avoiding repeated re-filtering of the same recording.
    """

    def __init__(
        self,
        patient_id: str,
        data_dir: str,
        windows: List[WindowRecord],
        file_records: List[FileRecord],
        raw_cache_size: int = 2,
    ):
        self.patient_id = patient_id
        self.data_dir = data_dir
        self.windows = windows
        self.file_records = file_records
        self.raw_cache_size = raw_cache_size
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def _get_file_data(self, file_idx: int) -> np.ndarray:
        if file_idx in self._cache:
            self._cache.move_to_end(file_idx)
            return self._cache[file_idx]

        data = load_bipolar_preprocessed(self.file_records[file_idx].filepath)
        self._cache[file_idx] = data
        if len(self._cache) > self.raw_cache_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        w = self.windows[idx]
        data = self._get_file_data(w.file_idx)
        segment = data[:, w.start_sample : w.start_sample + N_SAMPLES]
        if segment.shape[1] != N_SAMPLES:
            raise RuntimeError(
                f"Window at file_idx={w.file_idx}, start_sample={w.start_sample} is "
                f"truncated ({segment.shape[1]} < {N_SAMPLES} samples); the patient "
                "timeline is inconsistent with the underlying EDF file."
            )
        return torch.from_numpy(segment.copy()), w.label


class TimestampedWindowDataset(Dataset):
    """Wraps a CHBMITChronologicalDataset to also yield each window's
    abs_start_sec: (x, label) -> (x, label, abs_start_sec).

    CHBMITChronologicalDataset's own contract deliberately stays a plain
    (x, label) pair (the natural shape for a training DataLoader), but
    anything that needs chronological order for its own sake -- FPR/h over
    continuous hours, plotting a metric against time -- needs the
    timestamp too. Shared by evaluate.py and any other script (e.g. an
    ablation baseline) that evaluates a fold's test set the same way.
    """

    def __init__(self, base: CHBMITChronologicalDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        w = self.base.windows[idx]
        return x, y, w.abs_start_sec


# --------------------------------------------------------------------------
# Leave-One-Seizure-Out splitting
# --------------------------------------------------------------------------


def get_loso_splits(
    patient_id: str,
    data_dir: str,
    test_seizure_idx: int,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    num_workers: int = 0,
    raw_cache_size: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build LOSO train/val/test DataLoaders for one held-out seizure.

    Test set: the held-out seizure's preictal windows plus the maximal
    contiguous run of interictal windows immediately preceding them --
    together one continuous chronological span ending exactly at that
    seizure's Seizure Prediction Horizon, with no shuffling.

    Train/val: every remaining window (all other seizures' preictal
    windows, and all interictal windows not used by the test span), sorted
    chronologically and split so validation is the most recent
    ``val_fraction`` of that pool -- val and test are both temporally
    ordered and never interleaved with training windows.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")

    file_records, windows, all_seizures = build_patient_windows(patient_id, data_dir)
    if not (0 <= test_seizure_idx < len(all_seizures)):
        raise ValueError(
            f"test_seizure_idx={test_seizure_idx} out of range; patient {patient_id} "
            f"has {len(all_seizures)} seizures."
        )

    test_preictal = [w for w in windows if w.label == LABEL_PREICTAL and w.seizure_idx == test_seizure_idx]
    onset, _offset = all_seizures[test_seizure_idx]
    preictal_span_start = onset - PREICTAL_START_MIN * 60.0
    boundary_abs = min((w.abs_start_sec for w in test_preictal), default=preictal_span_start)

    interictal_before = sorted(
        (w for w in windows if w.label == LABEL_INTERICTAL and w.abs_start_sec < boundary_abs),
        key=lambda w: w.abs_start_sec,
    )
    test_interictal: List[WindowRecord] = []
    expected_next_start = boundary_abs
    for w in reversed(interictal_before):
        if expected_next_start - w.abs_start_sec <= WINDOW_SEC + 1e-6:
            test_interictal.append(w)
            expected_next_start = w.abs_start_sec
        else:
            break
    test_interictal.reverse()

    test_windows = test_interictal + sorted(test_preictal, key=lambda w: w.abs_start_sec)
    test_keys = {(w.file_idx, w.start_sample) for w in test_windows}

    remaining = [w for w in windows if (w.file_idx, w.start_sample) not in test_keys]
    remaining.sort(key=lambda w: w.abs_start_sec)
    n_val = max(1, round(len(remaining) * val_fraction)) if remaining else 0
    val_windows = remaining[len(remaining) - n_val :] if n_val else []
    train_windows = remaining[: len(remaining) - n_val] if n_val else remaining

    if not train_windows or not val_windows or not test_windows:
        raise RuntimeError(
            f"LOSO split for {patient_id}, test_seizure_idx={test_seizure_idx} produced an "
            f"empty split (train={len(train_windows)}, val={len(val_windows)}, "
            f"test={len(test_windows)}); the patient's recording is too sparse relative to "
            "the interictal/preictal window definitions for this fold."
        )

    train_ds = CHBMITChronologicalDataset(patient_id, data_dir, train_windows, file_records, raw_cache_size)
    val_ds = CHBMITChronologicalDataset(patient_id, data_dir, val_windows, file_records, raw_cache_size)
    test_ds = CHBMITChronologicalDataset(patient_id, data_dir, test_windows, file_records, raw_cache_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
