"""NeuroLens evaluation: clinical, uncertainty, and XAI-faithfulness benchmark.

Reads the `training_summary.json` produced by train.py, re-loads every
fold's checkpoint (+ exported FAISS trajectory index), runs MC-Dropout
inference on that fold's held-out continuous chronological test span, and
reports:

    1. Confusion matrix, window- and event-level Sensitivity, and False
       Prediction Rate per hour (FPR/h), computed over continuous
       interictal hours actually evaluated.
    2. A reliability (calibration) curve and Brier score, both before and
       after the fold's fitted temperature scaling.
    3. Faithfulness (risk drop from the computed counterfactual delta_z vs.
       a magnitude-matched random Gaussian direction, with a paired
       Wilcoxon signed-rank test) and Bifurcation Point Error (BPE) on
       synthetic Jansen-Rit data, using that fold's own FAISS index.
    4. A single exportable JSON report with per-fold and pooled/aggregate
       results.

Usage:
    python evaluate.py --data_dir /path/to/chb-mit --checkpoint_dir checkpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow `python path/to/evaluate.py` from anywhere

from dataset import N_SAMPLES, SPH_MIN, TimestampedWindowDataset, WINDOW_SEC, get_loso_splits
from model import NeuroLensBackbone
from synthetic_nmm import generate_synthetic_bifurcation_dataset
from xai_engine import CounterfactualSteeringEngine, TrajectoryVectorDB


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class EvalConfig:
    data_dir: str
    checkpoint_dir: str
    device: str = "auto"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0

    mc_samples: int = 30
    decision_threshold: float = 0.5
    refractory_sec: float = 300.0  # alarm lockout window for event-level FPR/h
    n_calibration_bins: int = 15

    cf_mc_samples_grad: int = 5
    faithfulness_n_samples: int = 20
    faithfulness_target_risk: float = 0.1
    faithfulness_random_repeats: int = 5
    faithfulness_pgd_lr: float = 0.01
    faithfulness_pgd_max_iter: int = 100

    run_bifurcation_benchmark: bool = True
    synthetic_duration_min: float = 15.0
    synthetic_transition_start_min: float = 10.0
    synthetic_transition_duration_min: float = 0.5
    synthetic_n_trials: int = 2
    synthetic_seed0: int = 1000
    spike_n_std: float = 3.0
    min_sustain_windows: int = 3

    max_folds: Optional[int] = None


# --------------------------------------------------------------------------
# MC-Dropout inference over a fold's test set
# --------------------------------------------------------------------------


@torch.no_grad()
def run_mc_inference(
    model: NeuroLensBackbone, loader: DataLoader, device: torch.device, num_samples: int, temperature: float
) -> Dict[str, np.ndarray]:
    model.eval()
    probs_raw, probs_cal, labels, timestamps, entropy, mc_std, latents = [], [], [], [], [], [], []
    for x, y, ts in loader:
        x = x.to(device)
        mc = model.get_mc_prediction(x, num_samples=num_samples)
        mean_prob = mc["mean_prob"].clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(mean_prob / (1.0 - mean_prob))
        cal_prob = torch.sigmoid(logit / temperature)

        probs_raw.append(mean_prob.cpu().numpy())
        probs_cal.append(cal_prob.cpu().numpy())
        labels.append(y.numpy())
        timestamps.append(ts.numpy())
        entropy.append(mc["predictive_entropy"].cpu().numpy())
        mc_std.append(mc["mc_std"].cpu().numpy())
        # A single stochastic forward pass's latent vector, kept for
        # generate_paper_artifacts.py's 2D latent-trajectory map (Fig. 1) --
        # a representative embedding per window is enough for that
        # visualization; it is not used for any uncertainty-sensitive metric.
        with torch.no_grad():
            latents.append(model(x)["latent_vector"].cpu().numpy())

    return {
        "probs_raw": np.concatenate(probs_raw),
        "probs_calibrated": np.concatenate(probs_cal),
        "labels": np.concatenate(labels),
        "timestamps": np.concatenate(timestamps),
        "predictive_entropy": np.concatenate(entropy),
        "mc_std": np.concatenate(mc_std),
        "latent_vectors": np.concatenate(latents),
    }


# --------------------------------------------------------------------------
# 1. Clinical metrics
# --------------------------------------------------------------------------


def clinical_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    refractory_sec: float,
    window_sec: float = float(WINDOW_SEC),
) -> Dict:
    """Confusion matrix, window- and event-level sensitivity/FPR-per-hour.

    FPR/h is computed over the interictal hours actually present in this
    (continuous, chronologically ordered) test span. Event-level FPR/h
    groups interictal false positives into alarms with a `refractory_sec`
    lockout (an alarm suppresses further counted alarms until that many
    seconds have elapsed since it fired), matching how a deployed alarm
    system would behave; window-level FPR/h is the raw per-epoch rate.
    """
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    interictal_mask = labels == 0
    n_interictal_windows = int(interictal_mask.sum())
    interictal_hours = n_interictal_windows * window_sec / 3600.0

    fp_mask = interictal_mask & (preds == 1)
    n_fp_windows = int(fp_mask.sum())
    fpr_per_hour_window = float(n_fp_windows / interictal_hours) if interictal_hours > 0 else float("nan")

    fp_times = np.sort(timestamps[fp_mask])
    n_fp_events = 0
    last_alarm = -np.inf
    for t in fp_times:
        if t - last_alarm > refractory_sec:
            n_fp_events += 1
            last_alarm = t  # suppressed detections within the lockout do NOT reset the timer
    fpr_per_hour_event = float(n_fp_events / interictal_hours) if interictal_hours > 0 else float("nan")

    auc_roc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else None
    auprc = float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else None

    return {
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sensitivity_window": sensitivity,
        "specificity": specificity,
        "interictal_hours_evaluated": interictal_hours,
        "fpr_per_hour_window": fpr_per_hour_window,
        "fpr_per_hour_event": fpr_per_hour_event,
        "n_false_positive_events": n_fp_events,
        "auc_roc": auc_roc,
        "auprc": auprc,
    }


# --------------------------------------------------------------------------
# 2. Calibration
# --------------------------------------------------------------------------


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[float, List[Dict]]:
    """Standard equal-width-bin ECE (Guo et al., 2017); also returns the
    per-bin (confidence, empirical accuracy, count) reliability-curve data."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0, "confidence": None, "accuracy": None})
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
        bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": count, "confidence": confidence, "accuracy": accuracy})
    return float(ece), bins


def uncertainty_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict:
    if probs.size == 0:
        return {"brier_score": None, "ece": None, "calibration_curve": [], "n_bins": n_bins}
    brier = float(brier_score_loss(labels, probs))
    ece, curve = expected_calibration_error(probs, labels, n_bins=n_bins)
    return {"brier_score": brier, "ece": ece, "calibration_curve": curve, "n_bins": n_bins}


# --------------------------------------------------------------------------
# 3a. Faithfulness via latent counterfactual perturbation
# --------------------------------------------------------------------------


def _matched_random_perturbation_risk(
    model: NeuroLensBackbone, z0: torch.Tensor, delta_z: torch.Tensor, n_repeats: int
) -> torch.Tensor:
    """Mean risk after nudging z0 by `n_repeats` random Gaussian directions,
    each rescaled to exactly ||delta_z||, so the comparison against the
    counterfactual perturbation isolates *direction*, not magnitude."""
    model.eval()
    norm = delta_z.norm().clamp_min(1e-12)
    risks = []
    with torch.no_grad():
        for _ in range(n_repeats):
            noise = torch.randn_like(delta_z)
            noise = noise / noise.norm().clamp_min(1e-12) * norm
            z_pert = F.normalize(z0 + noise, p=2, dim=-1)
            risk = torch.sigmoid(model.classifier(z_pert.unsqueeze(0)).squeeze(-1))
            risks.append(risk.squeeze())
    return torch.stack(risks).mean()


def faithfulness_benchmark(
    model: NeuroLensBackbone,
    cf_engine: CounterfactualSteeringEngine,
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int,
    target_risk: float,
    n_random_repeats: int,
    pgd_lr: float,
    pgd_max_iter: int,
) -> Dict:
    """For the highest-predicted-risk test windows, compares the risk
    reduction achieved by the computed counterfactual delta_z against a
    magnitude-matched random Gaussian direction. A faithful counterfactual
    engine should reduce risk significantly more than random chance."""
    model.eval()
    all_x, all_probs = [], []
    for x, _y in test_loader:
        with torch.no_grad():
            probs = torch.sigmoid(model(x.to(device))["seizure_logits"]).cpu()
        all_x.append(x)
        all_probs.append(probs)

    if not all_x:
        return {"n_samples": 0}

    x_cat = torch.cat(all_x, dim=0)
    probs_cat = torch.cat(all_probs, dim=0)
    n_use = min(n_samples, x_cat.shape[0])
    top_idx = torch.argsort(probs_cat, descending=True)[:n_use]

    initial_risks, cf_final_risks, random_mean_risks, delta_norms = [], [], [], []
    delta_tangents = []
    for idx in top_idx.tolist():
        x_win = x_cat[idx : idx + 1].to(device)
        with torch.no_grad():
            z0 = model(x_win)["latent_vector"][0]
        # raw_window enables the exact VJP pullback to tangent space (see
        # CounterfactualSteeringEngine._pullback_delta_to_tangent), which is
        # what makes the channel-level attribution below possible.
        result = cf_engine.compute_counterfactual_perturbation(
            model, z0, target_risk=target_risk, lr=pgd_lr, max_iter=pgd_max_iter, raw_window=x_win[0]
        )
        random_risk = _matched_random_perturbation_risk(model, z0, result.delta_z, n_random_repeats)

        initial_risks.append(result.initial_risk.item())
        cf_final_risks.append(result.final_risk.item())
        random_mean_risks.append(random_risk.item())
        delta_norms.append(result.delta_z.norm().item())
        if result.delta_tangent is not None:
            delta_tangents.append(result.delta_tangent.squeeze(0).detach().cpu().numpy())

    cf_arr = np.array(cf_final_risks)
    rand_arr = np.array(random_mean_risks)
    gains = rand_arr - cf_arr  # > 0: counterfactual direction beats matched-magnitude random noise

    p_value = None
    if len(gains) >= 2 and not np.allclose(cf_arr, rand_arr):
        try:
            p_value = float(wilcoxon(cf_arr, rand_arr).pvalue)
        except ValueError:
            p_value = None

    channel_attribution = None
    if delta_tangents:
        mean_delta_tangent = torch.from_numpy(np.mean(delta_tangents, axis=0)).float()
        try:
            proj = cf_engine.project_perturbation_to_channels(model.encoder.tangent_projector, mean_delta_tangent)
            channel_attribution = {
                "channel_power_delta": proj["channel_power_delta"],
                "region_delta": proj["region_delta"],
                "top_channel_pairs": [[a, b, float(v)] for a, b, v in proj["top_channel_pairs"]],
                "dominant_region": proj["dominant_region"],
                "dominant_direction": proj["dominant_direction"],
                "narrative": proj["narrative"],
            }
        except ValueError:
            channel_attribution = None

    return {
        "n_samples": int(n_use),
        "target_risk": target_risk,
        "mean_initial_risk": float(np.mean(initial_risks)),
        "mean_counterfactual_final_risk": float(cf_arr.mean()),
        "mean_random_final_risk": float(rand_arr.mean()),
        "mean_faithfulness_gain": float(gains.mean()),
        "std_faithfulness_gain": float(gains.std()),
        "mean_delta_z_norm": float(np.mean(delta_norms)),
        "wilcoxon_p_value": p_value,
        # Mean tangent-space delta over the sampled high-risk windows,
        # projected onto the 18-channel bipolar montage: which channels'
        # power/synchrony the counterfactual steering leans on most, for
        # generate_paper_artifacts.py's channel-level heatmap (Fig. 5).
        "channel_attribution": channel_attribution,
    }


# --------------------------------------------------------------------------
# 3b. Bifurcation Point Error on synthetic Jansen-Rit data
# --------------------------------------------------------------------------


def embed_eeg_windows(
    model: NeuroLensBackbone, eeg_windows: np.ndarray, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    """eeg_windows: [n_windows, C, N_SAMPLES] -> [n_windows, latent_dim], via
    the model's own forward pass. Shared by the synthetic bifurcation
    benchmark below and by artifact_robustness.py, which both need to embed
    an arbitrary raw EEG array (not one drawn from CHBMITChronologicalDataset)."""
    model.eval()
    if eeg_windows.shape[0] == 0:
        return np.zeros((0, model.latent_dim), dtype=np.float32)
    embeddings = []
    with torch.no_grad():
        for start in range(0, eeg_windows.shape[0], batch_size):
            xb = torch.from_numpy(eeg_windows[start : start + batch_size]).float().to(device)
            embeddings.append(model(xb)["latent_vector"].cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def compute_tas_curve(
    faiss_db: TrajectoryVectorDB, embeddings: np.ndarray, k: int, target_class: str = "preictal", top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling k-window trajectories over `embeddings` -> TAS(t) against
    `target_class`'s historical trajectories in `faiss_db`.

    Returns (tas_times_sec, tas_series); tas_times_sec[i] is the elapsed
    time (from the start of `embeddings`) of trajectory i's most recent
    ("current") window -- matching how TrajectoryVectorDB.build_index
    labels a whole trajectory by its own last window.
    """
    n_windows = embeddings.shape[0]
    if n_windows < k:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    tas_series, tas_times = [], []
    for start in range(0, n_windows - k + 1):
        traj = embeddings[start : start + k]
        matches = faiss_db.query_nearest_trajectories(traj, top_k=top_k)
        align = faiss_db.compute_trajectory_alignment_score(traj, matches)
        tas_series.append(align.tas_by_class.get(target_class, float("nan")))
        tas_times.append((start + k - 1) * WINDOW_SEC)
    return np.asarray(tas_times, dtype=np.float64), np.asarray(tas_series, dtype=np.float64)


def bifurcation_benchmark(
    model: NeuroLensBackbone,
    faiss_db: TrajectoryVectorDB,
    device: torch.device,
    duration_mins: float,
    transition_start_min: float,
    transition_duration_min: float,
    n_trials: int,
    seed0: int,
    spike_n_std: float,
    min_sustain_windows: int,
) -> Dict:
    """Embeds a synthetic Jansen-Rit recording (exact, analytically-located
    Hopf bifurcation onset from synthetic_nmm.py), computes TAS(t) of its
    rolling live trajectory against the fold's own historical PREICTAL
    trajectories, detects the first sustained spike above an
    adaptively-set (baseline-statistics) threshold, and reports the signed
    time delta to the true bifurcation onset.
    """
    k = faiss_db.k
    trials = []
    for trial in range(n_trials):
        seed = seed0 + trial
        synth = generate_synthetic_bifurcation_dataset(
            duration_mins=duration_mins,
            sampling_rate=256,
            n_channels=18,
            transition_start_min=transition_start_min,
            transition_duration_min=transition_duration_min,
            seed=seed,
        )
        eeg = synth.eeg_uv.numpy()  # [18, n_samples]
        n_windows = eeg.shape[1] // N_SAMPLES
        eeg = eeg[:, : n_windows * N_SAMPLES]
        eeg_windows = eeg.reshape(18, n_windows, N_SAMPLES).transpose(1, 0, 2)  # [n_windows, 18, N_SAMPLES]

        if n_windows < k:
            trials.append({"trial": trial, "seed": seed, "error": f"synthetic recording too short ({n_windows} windows < k={k})"})
            continue

        embeddings = embed_eeg_windows(model, eeg_windows, device)  # [n_windows, latent_dim]
        tas_times, tas_series = compute_tas_curve(faiss_db, embeddings, k, target_class="preictal")

        if np.all(np.isnan(tas_series)):
            true_onset_sec = float(synth.ground_truth.bifurcation_onset_sec)
            trials.append(
                {
                    "trial": trial,
                    "seed": seed,
                    "true_bifurcation_onset_sec": true_onset_sec,
                    "detected_onset_sec": None,
                    "bpe_sec": None,
                    "threshold": None,
                    "baseline_mean_tas": None,
                    "baseline_std_tas": None,
                    "n_windows": int(n_windows),
                    "detected": False,
                    "note": "no historical 'preictal' trajectories in this fold's FAISS index; TAS(t) is undefined",
                    "tas_times_sec": tas_times.tolist(),
                    "tas_series": tas_series.tolist(),
                }
            )
            continue

        baseline_mask = tas_times < transition_start_min * 60.0
        if baseline_mask.sum() < 3 or np.all(np.isnan(tas_series[baseline_mask])):
            baseline_mask = ~np.isnan(tas_series)
        baseline_mean = float(np.nanmean(tas_series[baseline_mask]))
        baseline_std = float(np.nanstd(tas_series[baseline_mask]))
        threshold = baseline_mean + spike_n_std * max(baseline_std, 1e-6)

        above = tas_series > threshold
        detected_idx = None
        run_len = 0
        for i, flag in enumerate(above):
            run_len = run_len + 1 if flag else 0
            if run_len >= min_sustain_windows:
                detected_idx = i - min_sustain_windows + 1
                break

        true_onset_sec = float(synth.ground_truth.bifurcation_onset_sec)
        if detected_idx is not None:
            detected_onset_sec = float(tas_times[detected_idx])
            bpe_sec = detected_onset_sec - true_onset_sec
        else:
            detected_onset_sec, bpe_sec = None, None

        trials.append(
            {
                "trial": trial,
                "seed": seed,
                "true_bifurcation_onset_sec": true_onset_sec,
                "detected_onset_sec": detected_onset_sec,
                "bpe_sec": bpe_sec,
                "threshold": threshold,
                "baseline_mean_tas": baseline_mean,
                "baseline_std_tas": baseline_std,
                "n_windows": int(n_windows),
                "detected": detected_idx is not None,
                # Full TAS(t) curve, kept for downstream plotting (e.g. TAS vs.
                # time-to-seizure in generate_paper_artifacts.py) rather than
                # only the summary statistics above.
                "tas_times_sec": tas_times.tolist(),
                "tas_series": tas_series.tolist(),
            }
        )

    valid_abs_bpe = [abs(t["bpe_sec"]) for t in trials if t.get("bpe_sec") is not None]
    return {
        "trajectory_k": k,
        "n_trials": n_trials,
        "n_detected": sum(1 for t in trials if t.get("detected")),
        "mean_abs_bpe_sec": float(np.mean(valid_abs_bpe)) if valid_abs_bpe else None,
        "std_abs_bpe_sec": float(np.std(valid_abs_bpe)) if valid_abs_bpe else None,
        "sph_sec": SPH_MIN * 60.0,  # Seizure Prediction Horizon boundary, for plotting
        "trials": trials,
    }


# --------------------------------------------------------------------------
# Per-fold orchestration
# --------------------------------------------------------------------------


def evaluate_fold(
    fold_entry: Dict, config: EvalConfig, device: torch.device, log: logging.Logger
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    patient_id = fold_entry["patient_id"]
    test_seizure_idx = fold_entry["test_seizure_idx"]
    fold_id = fold_entry["fold_id"]
    log.info(f"Evaluating fold {fold_id}")

    ckpt = torch.load(fold_entry["checkpoint"], map_location=device)
    model = NeuroLensBackbone(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    temperature = float(ckpt.get("temperature", 1.0))

    _, _, test_loader = get_loso_splits(
        patient_id, config.data_dir, test_seizure_idx, batch_size=config.batch_size, num_workers=config.num_workers
    )
    ts_loader = DataLoader(
        TimestampedWindowDataset(test_loader.dataset), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    mc = run_mc_inference(model, ts_loader, device, config.mc_samples, temperature)
    clinical = clinical_metrics(
        mc["probs_calibrated"], mc["labels"], mc["timestamps"], config.decision_threshold, config.refractory_sec
    )
    uncertainty_raw = uncertainty_metrics(mc["probs_raw"], mc["labels"], config.n_calibration_bins)
    uncertainty_calibrated = uncertainty_metrics(mc["probs_calibrated"], mc["labels"], config.n_calibration_bins)

    cf_engine = CounterfactualSteeringEngine(mc_samples_grad=config.cf_mc_samples_grad)
    try:
        faithfulness = faithfulness_benchmark(
            model,
            cf_engine,
            test_loader,
            device,
            n_samples=config.faithfulness_n_samples,
            target_risk=config.faithfulness_target_risk,
            n_random_repeats=config.faithfulness_random_repeats,
            pgd_lr=config.faithfulness_pgd_lr,
            pgd_max_iter=config.faithfulness_pgd_max_iter,
        )
    except Exception as exc:
        # A PGD/numerical hiccup here must not cost this fold's otherwise-
        # fine clinical and calibration metrics, computed above.
        log.exception(f"Faithfulness benchmark failed for fold {fold_id}: {exc}")
        faithfulness = {"n_samples": 0, "error": str(exc)}

    bifurcation = None
    if config.run_bifurcation_benchmark:
        index_dir = fold_entry.get("faiss_index_dir")
        if not index_dir:
            log.warning(f"Fold {fold_id} has no exported FAISS index; skipping bifurcation benchmark")
        else:
            try:
                db = TrajectoryVectorDB.load(index_dir)
                bifurcation = bifurcation_benchmark(
                    model,
                    db,
                    device,
                    duration_mins=config.synthetic_duration_min,
                    transition_start_min=config.synthetic_transition_start_min,
                    transition_duration_min=config.synthetic_transition_duration_min,
                    n_trials=config.synthetic_n_trials,
                    seed0=config.synthetic_seed0,
                    spike_n_std=config.spike_n_std,
                    min_sustain_windows=config.min_sustain_windows,
                )
            except Exception as exc:
                log.exception(f"Bifurcation benchmark failed for fold {fold_id}: {exc}")

    report = {
        "fold_id": fold_id,
        "patient_id": patient_id,
        "test_seizure_idx": test_seizure_idx,
        "n_test_windows": int(len(mc["labels"])),
        "val_auprc": fold_entry.get("val_auprc"),
        "temperature": temperature,
        "clinical": clinical,
        "uncertainty_raw": uncertainty_raw,
        "uncertainty_calibrated": uncertainty_calibrated,
        "faithfulness": faithfulness,
        "bifurcation": bifurcation,
        # Path to this fold's exported FAISS trajectory index (from
        # training_summary.json), so a downstream consumer can load its raw
        # trajectories.npy/labels.npy directly (no faiss import needed) for
        # generate_paper_artifacts.py's latent-trajectory map (Fig. 1).
        "faiss_index_dir": fold_entry.get("faiss_index_dir"),
        # Per-window predictions and latent embeddings, kept (not just the
        # summary "clinical"/"uncertainty" stats above) so a downstream
        # consumer -- e.g. run_ablations.py's ROC plot, or
        # generate_paper_artifacts.py's latent-trajectory map -- can rebuild
        # a full ROC/PR curve or a live-trajectory overlay without
        # re-running MC-Dropout inference.
        "probs_calibrated": mc["probs_calibrated"].tolist(),
        "labels": mc["labels"].tolist(),
        "timestamps": mc["timestamps"].tolist(),
        "latent_vectors": mc["latent_vectors"].tolist(),
    }
    raw = {"probs_calibrated": mc["probs_calibrated"], "probs_raw": mc["probs_raw"], "labels": mc["labels"]}
    return report, raw


def aggregate_reports(fold_reports: List[Dict], raw_arrays: List[Dict[str, np.ndarray]], config: EvalConfig) -> Dict:
    if not fold_reports:
        return {}

    pooled_probs = np.concatenate([r["probs_calibrated"] for r in raw_arrays])
    pooled_probs_raw = np.concatenate([r["probs_raw"] for r in raw_arrays])
    pooled_labels = np.concatenate([r["labels"] for r in raw_arrays])

    pooled_clinical = None
    if len(np.unique(pooled_labels)) > 1:
        pooled_clinical = {
            "auc_roc": float(roc_auc_score(pooled_labels, pooled_probs)),
            "auprc": float(average_precision_score(pooled_labels, pooled_probs)),
        }
    # Both pre- (raw) and post-temperature-scaling (calibrated) pooled
    # reliability curves, for generate_paper_artifacts.py's calibration
    # comparison figure (Fig. 4).
    pooled_uncertainty_raw = uncertainty_metrics(pooled_probs_raw, pooled_labels, config.n_calibration_bins)
    pooled_uncertainty = uncertainty_metrics(pooled_probs, pooled_labels, config.n_calibration_bins)

    event_caught = [f["clinical"]["confusion_matrix"]["tp"] > 0 for f in fold_reports]
    fpr_window_vals = [f["clinical"]["fpr_per_hour_window"] for f in fold_reports if not math.isnan(f["clinical"]["fpr_per_hour_window"])]
    fpr_event_vals = [f["clinical"]["fpr_per_hour_event"] for f in fold_reports if not math.isnan(f["clinical"]["fpr_per_hour_event"])]

    faithfulness_gains = [
        f["faithfulness"]["mean_faithfulness_gain"] for f in fold_reports if f.get("faithfulness", {}).get("n_samples", 0) > 0
    ]
    bpe_vals = [
        f["bifurcation"]["mean_abs_bpe_sec"]
        for f in fold_reports
        if f.get("bifurcation") and f["bifurcation"].get("mean_abs_bpe_sec") is not None
    ]

    return {
        "n_folds": len(fold_reports),
        "event_level_sensitivity": float(np.mean(event_caught)) if event_caught else None,
        "mean_fpr_per_hour_window": float(np.mean(fpr_window_vals)) if fpr_window_vals else None,
        "mean_fpr_per_hour_event": float(np.mean(fpr_event_vals)) if fpr_event_vals else None,
        "pooled_clinical": pooled_clinical,
        "pooled_uncertainty": pooled_uncertainty,
        "pooled_uncertainty_raw": pooled_uncertainty_raw,
        "mean_faithfulness_gain": float(np.mean(faithfulness_gains)) if faithfulness_gains else None,
        "std_faithfulness_gain": float(np.std(faithfulness_gains)) if faithfulness_gains else None,
        "mean_abs_bpe_sec_across_folds": float(np.mean(bpe_vals)) if bpe_vals else None,
        "std_abs_bpe_sec_across_folds": float(np.std(bpe_vals)) if bpe_vals else None,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing train.py's training_summary.json.")
    p.add_argument("--output_json", type=str, default=None, help="Defaults to <checkpoint_dir>/evaluation_report.json.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--decision_threshold", type=float, default=0.5)
    p.add_argument("--refractory_sec", type=float, default=300.0)
    p.add_argument("--n_calibration_bins", type=int, default=15)

    p.add_argument("--faithfulness_n_samples", type=int, default=20)
    p.add_argument("--faithfulness_target_risk", type=float, default=0.1)
    p.add_argument("--faithfulness_random_repeats", type=int, default=5)
    p.add_argument("--faithfulness_pgd_lr", type=float, default=0.01)
    p.add_argument("--faithfulness_pgd_max_iter", type=int, default=100)

    p.add_argument("--no_bifurcation_benchmark", action="store_true")
    p.add_argument("--synthetic_duration_min", type=float, default=15.0)
    p.add_argument("--synthetic_transition_start_min", type=float, default=10.0)
    p.add_argument("--synthetic_transition_duration_min", type=float, default=0.5)
    p.add_argument("--synthetic_n_trials", type=int, default=2)

    p.add_argument("--max_folds", type=int, default=None, help="Evaluate only the first N folds (useful for smoke tests).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("neurolens.evaluate")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    log.info(f"Using device: {device}")

    summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"{summary_path} not found; run train.py first.")
    with open(summary_path) as f:
        training_summary = json.load(f)
    folds = training_summary.get("folds", [])
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise RuntimeError(f"No trained folds found in {summary_path}")

    config = EvalConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=str(device),
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mc_samples=args.mc_samples,
        decision_threshold=args.decision_threshold,
        refractory_sec=args.refractory_sec,
        n_calibration_bins=args.n_calibration_bins,
        faithfulness_n_samples=args.faithfulness_n_samples,
        faithfulness_target_risk=args.faithfulness_target_risk,
        faithfulness_random_repeats=args.faithfulness_random_repeats,
        faithfulness_pgd_lr=args.faithfulness_pgd_lr,
        faithfulness_pgd_max_iter=args.faithfulness_pgd_max_iter,
        run_bifurcation_benchmark=not args.no_bifurcation_benchmark,
        synthetic_duration_min=args.synthetic_duration_min,
        synthetic_transition_start_min=args.synthetic_transition_start_min,
        synthetic_transition_duration_min=args.synthetic_transition_duration_min,
        synthetic_n_trials=args.synthetic_n_trials,
        max_folds=args.max_folds,
    )

    fold_reports: List[Dict] = []
    raw_arrays: List[Dict[str, np.ndarray]] = []
    for fold_entry in folds:
        try:
            report, raw = evaluate_fold(fold_entry, config, device, log)
        except Exception as exc:
            log.exception(f"Failed to evaluate fold {fold_entry.get('fold_id')}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        fold_reports.append(report)
        raw_arrays.append(raw)

    aggregate = aggregate_reports(fold_reports, raw_arrays, config)

    final_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "n_folds_evaluated": len(fold_reports),
        "folds": fold_reports,
        "aggregate": aggregate,
    }
    out_path = args.output_json or os.path.join(args.checkpoint_dir, "evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2, default=_json_default)
    log.info(f"Evaluation complete: {len(fold_reports)}/{len(folds)} fold(s). Report written to {out_path}")


if __name__ == "__main__":
    main()
