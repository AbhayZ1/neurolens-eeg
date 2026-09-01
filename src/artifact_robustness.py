"""Artifact-robustness stress test for the frozen NeuroLens FAISS engine.

CHB-MIT scalp EEG is well known to carry EMG (muscle) and EOG/movement
artifacts that can superficially resemble the high-frequency, non-
stationary character of preictal activity. This script takes a REAL,
clean, continuous interictal recording from CHB-MIT (not synthetic dummy
data), synthetically contaminates it at increasing severity with:

    - EMG-like broadband noise, band-limited to 20-45 Hz (the classic
      muscle-artifact band, deliberately overlapping the top of our own
      0.5-45 Hz preprocessing passband -- if this leaks the model), with a
      per-channel gain reflecting realistic clinical EMG susceptibility
      (strongest on temporal/frontal chains near the temporalis and
      frontalis muscles, weakest occipitally).
    - Baseline wander (simulated sweat/electrode-impedance drift and slow
      movement): a smoothed random walk, part shared across channels (a
      common-mode reference/movement effect) and part independent per
      channel (local contact variation).

The contaminated signal is then run through a *frozen* (eval-mode, no
gradient updates) trained NeuroLensBackbone + its exported FAISS
TrajectoryVectorDB, exactly as evaluate.py's bifurcation benchmark does,
to compute the live Trajectory Alignment Score TAS(t) against the fold's
historical PREICTAL trajectories. Since the underlying signal is genuinely
interictal throughout, ANY sustained rise of TAS(t) above a baseline-
derived threshold as contamination increases is by definition a false
positive: the retrieval mechanism mistaking a muscle/movement artifact for
a preictal trajectory. The classifier's own MC-Dropout risk is reported
alongside TAS as a second, independent robustness signal.

Usage:
    python artifact_robustness.py --data_dir /path/to/chb-mit --patient_id chb01 \
        --checkpoint checkpoints/chb01_seizure0.pt \
        --faiss_index_dir checkpoints/chb01_seizure0_faiss_index \
        --output_json robustness_chb01.json --output_plot robustness_chb01.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow `python path/to/artifact_robustness.py` from anywhere

from dataset import (
    BIPOLAR_MONTAGE,
    CHBMITChronologicalDataset,
    FS,
    LABEL_INTERICTAL,
    N_SAMPLES,
    WindowRecord,
    build_patient_windows,
    group_contiguous_runs,
)
from evaluate import compute_tas_curve, embed_eeg_windows
from model import NeuroLensBackbone
from xai_engine import TrajectoryVectorDB


# --------------------------------------------------------------------------
# Realistic per-channel EMG susceptibility (clinical double-banana chains
# nearest the temporalis/frontalis muscles pick up the most contamination;
# posterior/occipital chains the least). Must index-align with
# dataset.BIPOLAR_MONTAGE.
# --------------------------------------------------------------------------

_DEFAULT_EMG_CHANNEL_WEIGHT: Dict[str, float] = {
    "FP1-F7": 1.5, "F7-T7": 1.8, "T7-P7": 1.3, "P7-O1": 0.6,
    "FP1-F3": 1.2, "F3-C3": 0.9, "C3-P3": 0.7, "P3-O1": 0.5,
    "FP2-F4": 1.2, "F4-C4": 0.9, "C4-P4": 0.7, "P4-O2": 0.5,
    "FP2-F8": 1.5, "F8-T8": 1.8, "T8-P8": 1.3, "P8-O2": 0.6,
    "FZ-CZ": 0.8, "CZ-PZ": 0.6,
}


# --------------------------------------------------------------------------
# Artifact injection
# --------------------------------------------------------------------------


@dataclass
class ArtifactInjectionConfig:
    channel_names: List[str] = field(default_factory=lambda: list(BIPOLAR_MONTAGE))
    emg_band_hz: Tuple[float, float] = (20.0, 45.0)
    emg_amplitude_uv: float = 25.0        # per-channel noise std at severity=1.0, before per-channel weighting
    emg_channel_weights: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_EMG_CHANNEL_WEIGHT))
    wander_cutoff_hz: float = 0.5          # lowpass cutoff defining "slow" drift
    wander_amplitude_uv: float = 50.0      # std dev at severity=1.0
    wander_shared_fraction: float = 0.6    # fraction of wander that is common-mode across channels


@dataclass
class ContaminatedSignal:
    clean: np.ndarray             # [C, T]
    contaminated: np.ndarray      # [C, T]
    emg_component: np.ndarray     # [C, T]
    wander_component: np.ndarray  # [C, T]
    severity: float
    seed: int


def _zero_phase_filter(x: np.ndarray, fs: float, low: Optional[float], high: Optional[float], order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth filter along the last axis (matches dataset.py's
    own zero-phase, 4th-order Butterworth preprocessing convention). Give
    only `high` for a lowpass, only `low` for a highpass, or both for a
    bandpass."""
    nyq = fs / 2.0
    if low is not None and high is not None:
        sos = signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    elif high is not None:
        sos = signal.butter(order, high / nyq, btype="low", output="sos")
    elif low is not None:
        sos = signal.butter(order, low / nyq, btype="high", output="sos")
    else:
        raise ValueError("at least one of low/high must be given")
    return signal.sosfiltfilt(sos, x, axis=-1)


def inject_emg_noise(
    clean: np.ndarray, fs: float, config: ArtifactInjectionConfig, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Broadband EMG-like noise, band-limited to config.emg_band_hz, scaled
    per-channel by clinical EMG susceptibility. `severity` linearly scales
    amplitude (severity=1.0 -> config.emg_amplitude_uv std dev on a
    weight-1.0 channel)."""
    n_channels, n_samples = clean.shape
    white = rng.standard_normal((n_channels, n_samples))
    low, high = config.emg_band_hz
    band_limited = _zero_phase_filter(white, fs, low=low, high=high, order=4)
    band_limited /= np.clip(band_limited.std(axis=-1, keepdims=True), 1e-12, None)  # unit std per channel

    weights = np.array(
        [config.emg_channel_weights.get(name, 1.0) for name in config.channel_names], dtype=np.float64
    )
    if weights.shape[0] != n_channels:
        raise ValueError(f"config.channel_names has {weights.shape[0]} entries, clean signal has {n_channels} channels")
    amplitude = config.emg_amplitude_uv * severity
    return band_limited * (amplitude * weights)[:, None]


def inject_baseline_wander(
    clean: np.ndarray, fs: float, config: ArtifactInjectionConfig, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Slow baseline drift: a smoothed random walk, mixing a shared
    (common-mode, e.g. reference/movement) component with independent
    per-channel components, lowpass-filtered to `config.wander_cutoff_hz`."""
    n_channels, n_samples = clean.shape
    shared_walk = np.cumsum(rng.standard_normal(n_samples))
    independent_walk = np.cumsum(rng.standard_normal((n_channels, n_samples)), axis=-1)

    mixed = (
        config.wander_shared_fraction * shared_walk[None, :]
        + (1.0 - config.wander_shared_fraction) * independent_walk
    )
    smooth = _zero_phase_filter(mixed, fs, low=None, high=config.wander_cutoff_hz, order=4)
    smooth = smooth - smooth.mean(axis=-1, keepdims=True)
    smooth = smooth / np.clip(smooth.std(axis=-1, keepdims=True), 1e-12, None)

    amplitude = config.wander_amplitude_uv * severity
    return smooth * amplitude


def generate_contaminated_signal(
    clean: np.ndarray, fs: float, config: ArtifactInjectionConfig, severity: float, seed: int
) -> ContaminatedSignal:
    """clean: [C, T] microvolts (e.g. from dataset.load_bipolar_preprocessed)."""
    if severity < 0:
        raise ValueError(f"severity must be >= 0, got {severity}")
    rng = np.random.default_rng(seed)
    if severity == 0.0:
        emg = np.zeros_like(clean)
        wander = np.zeros_like(clean)
    else:
        emg = inject_emg_noise(clean, fs, config, severity, rng)
        wander = inject_baseline_wander(clean, fs, config, severity, rng)
    return ContaminatedSignal(
        clean=clean, contaminated=clean + emg + wander, emg_component=emg, wander_component=wander,
        severity=severity, seed=seed,
    )


# --------------------------------------------------------------------------
# Real clean interictal data
# --------------------------------------------------------------------------


def find_clean_interictal_run(patient_id: str, data_dir: str, min_windows: int) -> Tuple[np.ndarray, List[WindowRecord]]:
    """Locates the longest temporally-contiguous, fully-interictal-labeled
    real recording segment for `patient_id`, loads + preprocesses it via the
    exact same pipeline used for training (resample, zero-phase 4th-order
    Butterworth bandpass, notch), and returns it as one continuous
    [n_channels, n_windows * N_SAMPLES] microvolt array.
    """
    file_records, windows, _ = build_patient_windows(patient_id, data_dir)
    interictal = [w for w in windows if w.label == LABEL_INTERICTAL]
    if not interictal:
        raise RuntimeError(f"No interictal windows found for patient {patient_id}")

    runs = [r for r in group_contiguous_runs(interictal) if len(r) >= min_windows]
    if not runs:
        raise RuntimeError(
            f"No contiguous interictal run of at least {min_windows} windows "
            f"({min_windows * N_SAMPLES / FS:.0f}s) found for patient {patient_id}; "
            "try a smaller --min_windows or a different patient."
        )
    longest_run = max(runs, key=len)

    ds = CHBMITChronologicalDataset(patient_id, data_dir, longest_run, file_records)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    segments = [x.numpy() for x, _y in loader]
    window_array = np.concatenate(segments, axis=0)  # [n_windows, C, N_SAMPLES], chronological order preserved
    n_windows, n_channels, _ = window_array.shape
    continuous = window_array.transpose(1, 0, 2).reshape(n_channels, n_windows * N_SAMPLES)
    return continuous, longest_run


def windowize_signal(x: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    """[C, T] -> [n_windows, C, n_samples], dropping any trailing remainder.

    Public and model-agnostic (pure reshaping, no NeuroLens dependency) so
    other scripts -- e.g. run_ablations.py, comparing a non-NeuroLens
    baseline's response to the same contaminated signal -- can reuse it
    without going through this module's FAISS/TAS-specific machinery.
    """
    n_channels, n_total = x.shape
    n_windows = n_total // n_samples
    x = x[:, : n_windows * n_samples]
    return x.reshape(n_channels, n_windows, n_samples).transpose(1, 0, 2)


# --------------------------------------------------------------------------
# Robustness sweep
# --------------------------------------------------------------------------


@dataclass
class SeverityResult:
    severity: float
    tas_times_sec: np.ndarray
    tas_series: np.ndarray
    peak_tas: float
    sustained_spike: bool
    spike_onset_sec: Optional[float]
    risks: np.ndarray            # [n_windows], MC-Dropout mean seizure probability per window
    mean_risk: float
    peak_risk: float
    n_false_positive_windows: int  # windows where risk crosses `decision_threshold`


@dataclass
class RobustnessReport:
    patient_id: str
    n_clean_windows: int
    baseline_mean_tas: float
    baseline_std_tas: float
    threshold: float
    results: List[SeverityResult]

    @property
    def robust(self) -> bool:
        """True iff no contaminated (severity > 0) level produced a
        sustained TAS spike above the clean-signal baseline threshold."""
        return not any(r.sustained_spike for r in self.results if r.severity > 0)


@torch.no_grad()
def _batched_mc_risk(model: NeuroLensBackbone, windows: np.ndarray, device: torch.device, mc_samples: int, batch_size: int = 64) -> np.ndarray:
    model.eval()
    risks = []
    for start in range(0, windows.shape[0], batch_size):
        xb = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
        mc = model.get_mc_prediction(xb, num_samples=mc_samples)
        risks.append(mc["mean_prob"].cpu().numpy())
    return np.concatenate(risks) if risks else np.zeros(0, dtype=np.float32)


def run_robustness_sweep(
    model: NeuroLensBackbone,
    faiss_db: TrajectoryVectorDB,
    clean_signal: np.ndarray,
    fs: float,
    config: ArtifactInjectionConfig,
    device: torch.device,
    patient_id: str = "",
    severities: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    spike_n_std: float = 3.0,
    min_sustain_windows: int = 3,
    mc_samples: int = 20,
    seed: int = 0,
    decision_threshold: float = 0.5,
) -> RobustnessReport:
    """Runs the frozen model + FAISS engine over the clean signal and over
    each contamination severity, and checks for a sustained TAS(t) spike
    (>= min_sustain_windows consecutive windows above
    baseline_mean + spike_n_std * baseline_std, where the baseline is
    computed from the CLEAN signal alone) -- the same detection convention
    evaluate.py's synthetic bifurcation benchmark uses, applied here to
    data with no genuine regime change, so any detection is by construction
    a false positive.
    """
    k = faiss_db.k
    clean_windows = windowize_signal(clean_signal)
    n_windows = clean_windows.shape[0]
    if n_windows < k:
        raise RuntimeError(f"clean run too short: {n_windows} windows < FAISS index trajectory_k={k}")

    clean_embeddings = embed_eeg_windows(model, clean_windows, device)
    clean_times, clean_tas = compute_tas_curve(faiss_db, clean_embeddings, k, target_class="preictal")
    baseline_mean = float(np.nanmean(clean_tas))
    baseline_std = float(np.nanstd(clean_tas))
    threshold = baseline_mean + spike_n_std * max(baseline_std, 1e-6)

    def _evaluate(severity: float, tas_times: np.ndarray, tas_series: np.ndarray, windows: np.ndarray) -> SeverityResult:
        above = tas_series > threshold
        run_len = 0
        spike_idx = None
        for i, flag in enumerate(above):
            run_len = run_len + 1 if flag else 0
            if run_len >= min_sustain_windows:
                spike_idx = i - min_sustain_windows + 1
                break
        risks = _batched_mc_risk(model, windows, device, mc_samples)
        return SeverityResult(
            severity=severity,
            tas_times_sec=tas_times,
            tas_series=tas_series,
            peak_tas=float(np.nanmax(tas_series)) if tas_series.size else float("nan"),
            sustained_spike=spike_idx is not None,
            spike_onset_sec=float(tas_times[spike_idx]) if spike_idx is not None else None,
            risks=risks,
            mean_risk=float(np.mean(risks)) if risks.size else float("nan"),
            peak_risk=float(np.max(risks)) if risks.size else float("nan"),
            n_false_positive_windows=int(np.sum(risks > decision_threshold)),
        )

    results = [_evaluate(0.0, clean_times, clean_tas, clean_windows)]
    for severity in severities:
        cs = generate_contaminated_signal(clean_signal, fs, config, severity, seed=seed)
        windows = windowize_signal(cs.contaminated)
        embeddings = embed_eeg_windows(model, windows, device)
        tas_times, tas_series = compute_tas_curve(faiss_db, embeddings, k, target_class="preictal")
        results.append(_evaluate(severity, tas_times, tas_series, windows))

    return RobustnessReport(
        patient_id=patient_id,
        n_clean_windows=n_windows,
        baseline_mean_tas=baseline_mean,
        baseline_std_tas=baseline_std,
        threshold=threshold,
        results=results,
    )


def report_to_json_dict(report: RobustnessReport) -> Dict:
    return {
        "patient_id": report.patient_id,
        "n_clean_windows": report.n_clean_windows,
        "baseline_mean_tas": report.baseline_mean_tas,
        "baseline_std_tas": report.baseline_std_tas,
        "threshold": report.threshold,
        "robust": report.robust,
        "severities": [
            {
                "severity": r.severity,
                "peak_tas": r.peak_tas,
                "sustained_spike": r.sustained_spike,
                "spike_onset_sec": r.spike_onset_sec,
                "mean_risk": r.mean_risk,
                "peak_risk": r.peak_risk,
                "n_false_positive_windows": r.n_false_positive_windows,
                "n_windows": int(r.risks.size),
                "tas_times_sec": r.tas_times_sec.tolist(),
                "tas_series": r.tas_series.tolist(),
                "risks": r.risks.tolist(),
            }
            for r in report.results
        ],
    }


def plot_robustness_curves(report: RobustnessReport, output_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    n = len(report.results)
    for i, r in enumerate(report.results):
        color = cmap(i / max(n - 1, 1))
        label = "clean (severity=0)" if r.severity == 0.0 else f"severity={r.severity:g}"
        ax.plot(r.tas_times_sec, r.tas_series, color=color, label=label, linewidth=1.5, alpha=0.9)
        if r.spike_onset_sec is not None:
            ax.axvline(r.spike_onset_sec, color=color, linestyle=":", alpha=0.8)

    ax.axhline(
        report.threshold, color="crimson", linestyle="--", linewidth=1.5,
        label=f"spike threshold (clean mean + {(report.threshold - report.baseline_mean_tas):.3f})",
    )
    ax.set_xlabel("Time within clean interictal recording (s)")
    ax.set_ylabel("Trajectory Alignment Score (TAS) vs. historical preictal trajectories")
    ax.set_title(f"Artifact Robustness: TAS(t) under EMG + baseline-wander contamination\npatient {report.patient_id}")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--patient_id", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a train.py fold checkpoint (.pt)")
    p.add_argument("--faiss_index_dir", type=str, required=True, help="Path to that fold's exported FAISS index directory")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--output_plot", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--min_windows", type=int, default=None,
        help="Minimum contiguous clean-interictal windows required. Defaults to the FAISS index's "
        "trajectory_k plus 24 (2 min of margin), for a TAS(t) curve long enough to judge a 'sustained' spike.",
    )
    p.add_argument("--severities", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--spike_n_std", type=float, default=3.0)
    p.add_argument("--min_sustain_windows", type=int, default=3)
    p.add_argument("--mc_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--decision_threshold", type=float, default=0.5,
        help="Fixed clinical decision threshold for counting false-positive windows (independent of the "
        "statistical TAS spike threshold, which is derived from the clean signal itself).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("neurolens.artifact_robustness")

    device = _resolve_device(args.device)
    log.info(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = NeuroLensBackbone(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Loaded frozen checkpoint: {args.checkpoint}")

    faiss_db = TrajectoryVectorDB.load(args.faiss_index_dir)
    log.info(f"Loaded FAISS trajectory index: {args.faiss_index_dir} (k={faiss_db.k})")

    min_windows = args.min_windows if args.min_windows is not None else faiss_db.k + 24
    clean_signal, run_windows = find_clean_interictal_run(args.patient_id, args.data_dir, min_windows)
    n_windows = clean_signal.shape[1] // N_SAMPLES
    log.info(f"Found clean interictal run: {n_windows} windows ({n_windows * N_SAMPLES / FS / 60:.1f} min) for {args.patient_id}")

    config = ArtifactInjectionConfig()
    report = run_robustness_sweep(
        model,
        faiss_db,
        clean_signal,
        float(FS),
        config,
        device,
        patient_id=args.patient_id,
        severities=args.severities,
        spike_n_std=args.spike_n_std,
        min_sustain_windows=args.min_sustain_windows,
        mc_samples=args.mc_samples,
        seed=args.seed,
        decision_threshold=args.decision_threshold,
    )

    log.info(
        f"Clean-signal baseline TAS: mean={report.baseline_mean_tas:.4f} std={report.baseline_std_tas:.4f} "
        f"-> spike threshold={report.threshold:.4f}"
    )
    for r in report.results:
        tag = "[clean]        " if r.severity == 0.0 else f"[severity={r.severity:<5g}]"
        verdict = "SUSTAINED SPIKE -- ROBUSTNESS FAILURE" if r.sustained_spike else "stable -- pass"
        log.info(
            f"{tag} peak_tas={r.peak_tas:.4f} mean_risk={r.mean_risk:.4f} peak_risk={r.peak_risk:.4f} "
            f"fp_windows={r.n_false_positive_windows}/{r.risks.size} -> {verdict}"
        )
    log.info(f"OVERALL ROBUSTNESS VERDICT for {args.patient_id}: {'PASS' if report.robust else 'FAIL'}")

    output_json = args.output_json or f"artifact_robustness_{args.patient_id}.json"
    with open(output_json, "w") as f:
        json.dump(report_to_json_dict(report), f, indent=2)
    log.info(f"Report written to {output_json}")

    if args.output_plot:
        plot_robustness_curves(report, args.output_plot)
        log.info(f"Plot written to {args.output_plot}")

    return 0 if report.robust else 2  # distinct non-zero exit on a genuine robustness failure, for CI gating


if __name__ == "__main__":
    sys.exit(main())
