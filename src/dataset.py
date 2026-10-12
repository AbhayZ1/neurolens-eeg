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
import tempfile
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


_DATA_DIR_CACHE: Dict[Tuple[str, str], str] = {}


def resolve_data_dir(data_dir: str, patient_id: str, max_depth: int = 6) -> str:
    """Resolves a possibly-nested dataset root to the directory that
    directly contains `patient_id`'s subfolder.

    Kaggle "Add Data" mounts frequently wrap the actual database under an
    extra version-named directory, e.g.
    ``/kaggle/input/chb-mit-scalp-eeg-database/chb-mit-scalp-eeg-database-1.0.0/chb01/...``
    rather than ``chb01/`` sitting directly under the given root.

    Fast path: if ``data_dir/patient_id/{patient_id}-summary.txt`` already
    exists, `data_dir` is returned unchanged -- a no-op for an
    already-flat local layout, with no filesystem walk at all.
    Otherwise, searches recursively under `data_dir` (bounded to
    `max_depth` levels, so a large unrelated ``/kaggle/input`` mount can't
    trigger an unbounded scan) for a directory containing
    ``{patient_id}-summary.txt``, and returns its PARENT -- the directory
    that plays the same role as a flat `data_dir` for every function in
    this module. Results are cached per (data_dir, patient_id) so the walk
    only happens once even though every window-building call for that
    patient goes through here.
    """
    cache_key = (data_dir, patient_id)
    if cache_key in _DATA_DIR_CACHE:
        return _DATA_DIR_CACHE[cache_key]

    direct = os.path.join(data_dir, patient_id, f"{patient_id}-summary.txt")
    if os.path.isfile(direct):
        _DATA_DIR_CACHE[cache_key] = data_dir
        return data_dir

    target_name = f"{patient_id}-summary.txt"
    base_depth = os.path.normpath(data_dir).count(os.sep)
    for root, dirs, files in os.walk(data_dir):
        depth = os.path.normpath(root).count(os.sep) - base_depth
        if depth >= max_depth:
            dirs[:] = []  # prune: do not descend further from here
            continue
        if target_name in files:
            resolved = os.path.dirname(root)
            _DATA_DIR_CACHE[cache_key] = resolved
            return resolved

    raise FileNotFoundError(
        f"Could not find '{target_name}' anywhere under {data_dir} (searched up to "
        f"{max_depth} levels deep). Check that --data_dir points at, or contains at any "
        "nesting depth, the CHB-MIT database root."
    )


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
    data_dir = resolve_data_dir(data_dir, patient_id)
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


_CACHE_VERSION = "v1"


def _edf_cache_path(filepath: str) -> str:
    """On-disk location of one EDF's preprocessed array.

    Keyed on patient folder + file name + source size + every preprocessing
    constant, so changing the filter settings (or swapping in a different
    copy of the dataset) can never silently reuse a stale array.
    """
    root = os.environ.get("NEUROLENS_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "neurolens_edf_cache")
    stem = os.path.splitext(os.path.basename(filepath))[0]
    patient = os.path.basename(os.path.dirname(filepath))
    tag = f"{_CACHE_VERSION}_fs{FS}_bp{BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ}o{BANDPASS_ORDER}_n{NOTCH_FREQ_HZ}_sz{os.path.getsize(filepath)}"
    return os.path.join(root, patient, f"{stem}_{tag}.npy")


def load_bipolar_preprocessed(filepath: str, use_cache: bool = True) -> np.ndarray:
    """Load one EDF, enforce the 18-channel bipolar montage, and preprocess.

    Pipeline: pick + reorder to BIPOLAR_MONTAGE -> resample to FS Hz ->
    zero-phase 4th-order Butterworth bandpass (0.5-45 Hz) -> 60 Hz notch.

    Returns a (N_CHANNELS, n_samples) float32 array in microvolts.

    With ``use_cache`` (default) the result is written once to a per-file
    ``.npy`` (``$NEUROLENS_CACHE_DIR``, default: system temp dir) and every
    later call -- including from other DataLoader workers or other scripts
    -- returns a read-only memory map of it instead of re-reading and
    re-filtering ~1 hour of EEG (~5 s per file). The values are identical
    to the uncached path; only the repeated work is removed.
    """
    if not use_cache:
        return _preprocess_edf(filepath)

    cache_path = _edf_cache_path(filepath)
    if os.path.isfile(cache_path):
        return np.load(cache_path, mmap_mode="r")

    data = _preprocess_edf(filepath)
    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(tmp_path, "wb") as f:
            np.save(f, data)
        os.replace(tmp_path, cache_path)  # atomic: concurrent workers never see a half-written file
    except OSError:
        # Cache disk full/unwritable: fall back to the in-memory array rather than fail the run.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return data
    return np.load(cache_path, mmap_mode="r")


def _preprocess_edf(filepath: str) -> np.ndarray:
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
        return torch.from_numpy(np.array(segment)), w.label


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

    Test set: the held-out seizure's preictal windows plus the nearest
    available interictal windows chronologically before them (up to
    ``len(test_preictal)``, so the test set stays roughly class-balanced
    rather than becoming an unbounded interictal majority). These need not
    be second-by-second contiguous with the preictal span -- clinical_metrics()
    computes interictal-hours by counting interictal windows directly, not
    by measuring elapsed wall-clock time, so a gap (e.g. another seizure's
    exclusion buffer sitting between them) costs nothing downstream. An
    earlier version required strict contiguity, which left many folds with
    ZERO interictal test windows whenever the immediately-preceding 5s slot
    wasn't itself labeled interictal -- common whenever seizures are closely
    spaced, which silently made AUC-ROC/specificity/FPR-h undefined for
    that fold.

    Train/val: every remaining window (all other seizures' preictal
    windows, and all interictal windows not used by the test span). val is
    the most recent ``val_fraction`` of each class's own remaining pool,
    not of the pool as a whole -- splitting per class guarantees both
    classes appear in validation (needed for temperature-scaling and
    AUPRC-based model selection to run at all) even though the whole-pool
    timeline is preictal-sparse; it preserves the same "most recent slice
    is held out" anti-leakage principle, just applied within each class
    rather than across the mixed timeline where a long preictal run could
    otherwise dominate the tail and crowd out the other class entirely.
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
    n_test_interictal = max(len(test_preictal), 1)
    test_interictal = interictal_before[-n_test_interictal:] if interictal_before else []

    test_windows = test_interictal + sorted(test_preictal, key=lambda w: w.abs_start_sec)
    test_keys = {(w.file_idx, w.start_sample) for w in test_windows}

    remaining = [w for w in windows if (w.file_idx, w.start_sample) not in test_keys]

    def _tail_split(pool: List[WindowRecord]) -> Tuple[List[WindowRecord], List[WindowRecord]]:
        pool = sorted(pool, key=lambda w: w.abs_start_sec)
        n_val = round(len(pool) * val_fraction) if pool else 0
        if n_val == 0:
            return pool, []
        return pool[: len(pool) - n_val], pool[len(pool) - n_val :]

    train_interictal, val_interictal = _tail_split([w for w in remaining if w.label == LABEL_INTERICTAL])
    train_preictal, val_preictal = _tail_split([w for w in remaining if w.label == LABEL_PREICTAL])
    train_windows = sorted(train_interictal + train_preictal, key=lambda w: w.abs_start_sec)
    val_windows = sorted(val_interictal + val_preictal, key=lambda w: w.abs_start_sec)

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
