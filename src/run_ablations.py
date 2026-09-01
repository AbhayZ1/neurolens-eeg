"""Ablation orchestrator: ResNet1D baseline vs. NeuroLens.

Preempts the "how do we know your architecture is doing the work, not
just more parameters/more training" review by training a standard,
strong 1D-ResNet-18 classifier (baseline_model.py) on the *exact same*
LOSO chronological splits NeuroLens used, evaluating it with the *exact
same* clinical-metric code (evaluate.clinical_metrics), stress-testing
both models against the *exact same* injected-artifact realization
(artifact_robustness.py), and producing a direct head-to-head comparison.

Prerequisite: run_experiments.py (or train.py + evaluate.py) must already
have produced, under `--output_dir`:
    checkpoints/training_summary.json   -- which (patient, LOSO fold)
                                            pairs NeuroLens trained, and
                                            where its checkpoints/FAISS
                                            indices are.
    checkpoints/evaluation_report.json  -- NeuroLens's own clinical
                                            metrics + per-window
                                            probs/labels for those folds.
This script reproduces the baseline on the identical fold list found
there -- it does not re-derive splits independently -- so the comparison
in Phase 4 is guaranteed apples-to-apples.

Runs, in order, logging to `<output_dir>/ablation_run.log`:

    Phase 1 -- Baseline Training: dataset.get_loso_splits(...) (the same
               function, called with the same (patient, test_seizure_idx)
               pairs NeuroLens used) trains a fresh ResNet1D per fold with
               plain BCEWithLogitsLoss, keeping the best-val-AUPRC epoch
               (train.py's own model-selection convention, for fairness).
               Checkpoints saved to <output_dir>/baseline_checkpoints/.
    Phase 2 -- Baseline Evaluation: deterministic (no MC-Dropout -- the
               baseline has no epistemic-uncertainty layer by design)
               inference on each fold's held-out chronological test span;
               Sensitivity, FPR/h, and AUC-ROC computed via evaluate.py's
               own clinical_metrics -- the identical function NeuroLens's
               numbers were computed with.
    Phase 3 -- Robustness, at one worst-case EMG+wander severity:
               artifact_robustness.py's CLI is executed, unchanged, as a
               subprocess against each NeuroLens checkpoint + its FAISS
               index -- exactly as it was designed to run. The baseline
               has no latent trajectory space, so TAS is undefined for it
               and it structurally CANNOT run through that FAISS/TAS
               pipeline; instead this phase reuses artifact_robustness.py's
               model-agnostic signal utilities (find_clean_interictal_run,
               generate_contaminated_signal, windowize_signal) on the same
               clean recording and the same injected-noise seed, and
               measures the baseline's own sigmoid risk response -- the
               directly comparable "does contamination make this model
               cry wolf" signal for a plain classifier.
    Phase 4 -- Comparative artifacts: ablation_table.tex (Sensitivity,
               FPR/h, AUC-ROC, and False Positives under EMG noise, mean
               +/- 95% CI over folds) and roc_comparison.png (pooled ROC
               curves), built from NeuroLens's existing evaluation_report.json
               and this run's fresh baseline evaluation report.

Usage:
    python run_ablations.py --data_dir /path/to/chb-mit --output_dir runs/exp1 \
        --patients chb01 chb02 chb03 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow `python path/to/run_ablations.py` from anywhere

from artifact_robustness import (
    ArtifactInjectionConfig,
    find_clean_interictal_run,
    generate_contaminated_signal,
    windowize_signal,
)
from baseline_model import ResNet1D, resnet18_1d, resnet18_1d_lightweight
from dataset import FS, N_CHANNELS, TimestampedWindowDataset, get_loso_splits
from evaluate import clinical_metrics

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROBUSTNESS_SCRIPT = os.path.join(SRC_DIR, "artifact_robustness.py")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class AblationConfig:
    data_dir: str
    output_dir: str
    patients: Sequence[str]
    device: str = "auto"
    seed: int = 42

    # data loading -- must match train.py's for identical splits
    batch_size: int = 32
    val_fraction: float = 0.15
    num_workers: int = 0
    raw_cache_size: int = 2

    # Phase 1: baseline training
    epochs_baseline: int = 15
    lr_baseline: float = 1e-3
    baseline_dropout_p: float = 0.3
    lightweight_baseline: bool = False
    grad_clip_norm: float = 5.0

    # Phase 2: baseline evaluation
    decision_threshold: float = 0.5
    refractory_sec: float = 300.0

    # Phase 3: robustness
    worst_case_severity: float = 4.0
    robustness_min_windows: Optional[int] = None
    robustness_spike_n_std: float = 3.0
    robustness_min_sustain_windows: int = 3
    robustness_mc_samples: int = 20
    robustness_seed: int = 0
    skip_robustness: bool = False

    neurolens_checkpoint_dir: Optional[str] = None
    max_folds: Optional[int] = None


def _baseline_model_kwargs(config: AblationConfig) -> Dict:
    return dict(
        in_channels=N_CHANNELS,
        layers=(2, 2, 2, 2),
        base_width=32 if config.lightweight_baseline else 64,
        dropout_p=config.baseline_dropout_p,
        num_classes=1,
    )


def _build_baseline_model(config: AblationConfig) -> ResNet1D:
    factory = resnet18_1d_lightweight if config.lightweight_baseline else resnet18_1d
    return factory(in_channels=N_CHANNELS, dropout_p=config.baseline_dropout_p)


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("neurolens.ablations")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def _stream_subprocess(cmd: List[str], log: logging.Logger, stage_name: str) -> float:
    """Runs `cmd`, relaying its stdout/stderr into `log` line by line.
    artifact_robustness.py returns exit code 2 for a *successful run* that
    found a genuine robustness failure -- that is not itself an error here.
    """
    log.info(f"Launching subprocess for [{stage_name}]:")
    log.info("  " + " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        if line:
            log.info(f"[{stage_name}] {line}")
    process.wait()
    elapsed = time.time() - t0
    if process.returncode not in (0, 2):
        raise RuntimeError(f"[{stage_name}] subprocess exited with code {process.returncode} after {elapsed:.1f}s")
    log.info(f"[{stage_name}] completed in {elapsed:.1f}s (exit code {process.returncode})")
    return elapsed


# --------------------------------------------------------------------------
# Phase 1: baseline training
# --------------------------------------------------------------------------


def _train_epoch_baseline(
    model: ResNet1D, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, grad_clip_norm: float
) -> float:
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()
        logits = model(x)
        loss = bce(logits, y)
        loss.backward()
        if grad_clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def _quick_eval_auprc_baseline(model: ResNet1D, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        probs.append(torch.sigmoid(model(x)).cpu().numpy())
        labels.append(y.numpy())
    if not probs:
        return float("nan")
    probs_arr = np.concatenate(probs)
    labels_arr = np.concatenate(labels)
    if len(np.unique(labels_arr)) < 2:
        return float("nan")
    return float(average_precision_score(labels_arr, probs_arr))


@dataclass
class BaselineFoldResult:
    patient_id: str
    test_seizure_idx: int
    model: ResNet1D
    best_val_auprc: float
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    history: List[Dict]


def train_baseline_fold(
    patient_id: str, test_seizure_idx: int, config: AblationConfig, device: torch.device, log: logging.Logger
) -> BaselineFoldResult:
    train_loader, val_loader, test_loader = get_loso_splits(
        patient_id,
        config.data_dir,
        test_seizure_idx,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        num_workers=config.num_workers,
        raw_cache_size=config.raw_cache_size,
    )
    model = _build_baseline_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_baseline)

    best_auprc = float("-inf")
    best_state = None
    history: List[Dict] = []
    for epoch in range(config.epochs_baseline):
        train_loss = _train_epoch_baseline(model, train_loader, optimizer, device, config.grad_clip_norm)
        val_auprc = _quick_eval_auprc_baseline(model, val_loader, device)
        log.info(
            f"[baseline {patient_id} fold{test_seizure_idx}] epoch {epoch + 1}/{config.epochs_baseline} "
            f"train_loss={train_loss:.4f} val_auprc={val_auprc:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auprc": val_auprc})
        if not math.isnan(val_auprc) and val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        log.warning(f"[baseline {patient_id} fold{test_seizure_idx}] val_auprc never defined; keeping final-epoch weights")
        best_auprc = float("nan")

    return BaselineFoldResult(
        patient_id=patient_id,
        test_seizure_idx=test_seizure_idx,
        model=model,
        best_val_auprc=best_auprc,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        history=history,
    )


# --------------------------------------------------------------------------
# Phase 2: baseline evaluation
# --------------------------------------------------------------------------


@torch.no_grad()
def run_baseline_inference(model: ResNet1D, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    probs, labels, timestamps = [], [], []
    for x, y, ts in loader:
        x = x.to(device)
        probs.append(torch.sigmoid(model(x)).cpu().numpy())
        labels.append(y.numpy())
        timestamps.append(ts.numpy())
    return {
        "probs": np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32),
        "labels": np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64),
        "timestamps": np.concatenate(timestamps) if timestamps else np.zeros(0, dtype=np.float64),
    }


def evaluate_baseline_fold(
    model: ResNet1D,
    patient_id: str,
    test_seizure_idx: int,
    test_loader: DataLoader,
    config: AblationConfig,
    device: torch.device,
    log: logging.Logger,
) -> Dict:
    ts_loader = DataLoader(
        TimestampedWindowDataset(test_loader.dataset),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    out = run_baseline_inference(model, ts_loader, device)
    clinical = clinical_metrics(
        out["probs"], out["labels"], out["timestamps"], config.decision_threshold, config.refractory_sec
    )
    fold_id = f"{patient_id}_seizure{test_seizure_idx}"
    log.info(
        f"[baseline eval {fold_id}] sensitivity={clinical['sensitivity_window']:.4f} "
        f"fpr/h(event)={clinical['fpr_per_hour_event']:.4f} auc_roc={clinical['auc_roc']}"
    )
    return {
        "fold_id": fold_id,
        "patient_id": patient_id,
        "test_seizure_idx": test_seizure_idx,
        "n_test_windows": int(len(out["labels"])),
        "clinical": clinical,
        "probs": out["probs"].tolist(),
        "labels": out["labels"].tolist(),
    }


# --------------------------------------------------------------------------
# Phase 3: robustness
# --------------------------------------------------------------------------


def _read_faiss_k(faiss_index_dir: str) -> Optional[int]:
    manifest_path = os.path.join(faiss_index_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path) as f:
        return json.load(f).get("k")


def run_neurolens_robustness(
    patient_id: str, neurolens_fold_entry: Dict, min_windows: int, config: AblationConfig, log: logging.Logger
) -> Optional[Dict]:
    """Executes artifact_robustness.py, unchanged, as a subprocess against
    one NeuroLens checkpoint + its exported FAISS index."""
    fold_id = neurolens_fold_entry["fold_id"]
    faiss_dir = neurolens_fold_entry.get("faiss_index_dir")
    if not faiss_dir or not os.path.isdir(faiss_dir):
        log.warning(f"[robustness] no FAISS index for NeuroLens fold {fold_id}; skipping")
        return None

    ablation_dir = os.path.join(config.output_dir, "ablation_artifacts")
    os.makedirs(ablation_dir, exist_ok=True)
    out_json = os.path.join(ablation_dir, f"robustness_neurolens_{fold_id}.json")

    cmd = [
        sys.executable,
        ARTIFACT_ROBUSTNESS_SCRIPT,
        "--data_dir", config.data_dir,
        "--patient_id", patient_id,
        "--checkpoint", neurolens_fold_entry["checkpoint"],
        "--faiss_index_dir", faiss_dir,
        "--output_json", out_json,
        "--device", config.device,
        "--min_windows", str(min_windows),
        "--severities", str(config.worst_case_severity),
        "--spike_n_std", str(config.robustness_spike_n_std),
        "--min_sustain_windows", str(config.robustness_min_sustain_windows),
        "--mc_samples", str(config.robustness_mc_samples),
        "--seed", str(config.robustness_seed),
        "--decision_threshold", str(config.decision_threshold),
    ]
    try:
        _stream_subprocess(cmd, log, f"artifact_robustness[neurolens:{fold_id}]")
    except RuntimeError as exc:
        log.warning(f"[robustness] artifact_robustness.py failed for NeuroLens fold {fold_id}: {exc}")
        return None

    with open(out_json) as f:
        return json.load(f)


@torch.no_grad()
def _baseline_risk(model: ResNet1D, windows: np.ndarray, device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    risks = []
    for start in range(0, windows.shape[0], batch_size):
        xb = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
        risks.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(risks) if risks else np.zeros(0, dtype=np.float32)


_CLEAN_SIGNAL_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, int]] = {}


def run_baseline_robustness(
    model: ResNet1D, patient_id: str, min_windows: int, config: AblationConfig, device: torch.device, log: logging.Logger
) -> Optional[Dict]:
    """The baseline has no latent trajectory space, so TAS/FAISS retrieval
    is undefined for it -- artifact_robustness.py's pipeline cannot run
    against it. This measures the same clean-signal + same worst-case
    contamination realization's effect on its raw sigmoid risk instead,
    reusing artifact_robustness.py's model-agnostic signal utilities.
    """
    cache_key = (patient_id, min_windows)
    if cache_key not in _CLEAN_SIGNAL_CACHE:
        try:
            clean_signal, _run_windows = find_clean_interictal_run(patient_id, config.data_dir, min_windows)
        except RuntimeError as exc:
            log.warning(f"[robustness] no clean interictal run for {patient_id}: {exc}")
            return None
        _CLEAN_SIGNAL_CACHE[cache_key] = (clean_signal, clean_signal.shape[1])
    clean_signal, _ = _CLEAN_SIGNAL_CACHE[cache_key]

    inj_config = ArtifactInjectionConfig()
    clean_windows = windowize_signal(clean_signal)
    clean_risks = _baseline_risk(model, clean_windows, device)

    cs = generate_contaminated_signal(clean_signal, float(FS), inj_config, config.worst_case_severity, seed=config.robustness_seed)
    contaminated_windows = windowize_signal(cs.contaminated)
    contaminated_risks = _baseline_risk(model, contaminated_windows, device)

    n_fp_clean = int(np.sum(clean_risks > config.decision_threshold))
    n_fp_contaminated = int(np.sum(contaminated_risks > config.decision_threshold))

    log.info(
        f"[baseline robustness {patient_id}] clean fp={n_fp_clean}/{clean_risks.size} "
        f"contaminated(severity={config.worst_case_severity:g}) fp={n_fp_contaminated}/{contaminated_risks.size}"
    )
    return {
        "patient_id": patient_id,
        "severity": config.worst_case_severity,
        "n_windows": int(contaminated_risks.size),
        "mean_risk_clean": float(np.mean(clean_risks)) if clean_risks.size else float("nan"),
        "peak_risk_clean": float(np.max(clean_risks)) if clean_risks.size else float("nan"),
        "mean_risk_contaminated": float(np.mean(contaminated_risks)) if contaminated_risks.size else float("nan"),
        "peak_risk_contaminated": float(np.max(contaminated_risks)) if contaminated_risks.size else float("nan"),
        "n_false_positive_windows_clean": n_fp_clean,
        "n_false_positive_windows_contaminated": n_fp_contaminated,
    }


# --------------------------------------------------------------------------
# Phase 4: comparative artifacts
# --------------------------------------------------------------------------


def mean_confidence_interval(values: Sequence[Optional[float]], confidence: float = 0.95) -> Dict[str, Optional[float]]:
    """Mean and a Student's-t-based CI, appropriate for the small number of
    LOSO folds typically available here."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    n = arr.size
    if n == 0:
        return {"mean": None, "half_width": None, "n": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "half_width": None, "n": 1}
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    return {"mean": mean, "half_width": t_crit * sem, "n": n}


def _fmt(stat: Dict, scale: float = 1.0, decimals: int = 3) -> str:
    if stat["mean"] is None:
        return "N/A"
    mean = stat["mean"] * scale
    if stat["half_width"] is None:
        return f"{mean:.{decimals}f}"
    hw = stat["half_width"] * scale
    return f"{mean:.{decimals}f} $\\pm$ {hw:.{decimals}f}"


def _fp_rate_neurolens(robustness_entry: Optional[Dict]) -> Optional[float]:
    """Fraction of windows flagged false-positive at the (single) worst-case
    severity requested -- the last entry in `severities` (index 0 is always
    the clean/severity=0.0 baseline)."""
    if robustness_entry is None:
        return None
    severities = robustness_entry.get("severities", [])
    if len(severities) < 2:
        return None
    worst = severities[-1]
    n = worst.get("n_windows", 0)
    return (worst["n_false_positive_windows"] / n) if n > 0 else None


def _fp_rate_baseline(robustness_entry: Optional[Dict]) -> Optional[float]:
    if robustness_entry is None:
        return None
    n = robustness_entry.get("n_windows", 0)
    return (robustness_entry["n_false_positive_windows_contaminated"] / n) if n > 0 else None


def build_ablation_table(
    neurolens_folds: List[Dict],
    baseline_folds: List[Dict],
    neurolens_robustness_by_fold: Dict[str, Dict],
    baseline_robustness_by_fold: Dict[str, Dict],
    worst_case_severity: float,
    output_path: str,
) -> str:
    def _row(label: str, fold_list: List[Dict], robustness_by_fold: Dict[str, Dict], fp_extractor) -> str:
        sens = mean_confidence_interval([f["clinical"]["sensitivity_window"] for f in fold_list])
        fpr = mean_confidence_interval([f["clinical"]["fpr_per_hour_event"] for f in fold_list])
        auc = mean_confidence_interval([f["clinical"]["auc_roc"] for f in fold_list])
        fp_rate = mean_confidence_interval([fp_extractor(robustness_by_fold.get(f["fold_id"])) for f in fold_list])
        return (
            f"{label} & {_fmt(sens, scale=100.0, decimals=1)} & {_fmt(fpr, decimals=3)} & "
            f"{_fmt(auc, decimals=3)} & {_fmt(fp_rate, scale=100.0, decimals=1)} \\\\"
        )

    nl_row = _row("NeuroLens", neurolens_folds, neurolens_robustness_by_fold, _fp_rate_neurolens)
    bl_row = _row("ResNet1D (baseline)", baseline_folds, baseline_robustness_by_fold, _fp_rate_baseline)

    lines = [
        "% Auto-generated by run_ablations.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{NeuroLens vs. a standard 1D-ResNet-18 baseline, trained and evaluated on identical "
        "Leave-One-Seizure-Out chronological splits with identical Sensitivity/FPR-per-hour/AUC-ROC "
        "computation (mean $\\pm$ 95\\% CI over LOSO folds). "
        f"``FP under EMG noise'' is the fraction of a real, clean interictal recording's windows "
        f"misclassified as preictal after injecting worst-case (severity={worst_case_severity:g}) EMG "
        "muscle-artifact and baseline-wander contamination; for NeuroLens this is its classifier-risk "
        "signal (reported alongside, but distinct from, its trajectory-retrieval TAS robustness check, "
        "which has no baseline analogue since the baseline has no latent trajectory representation).}",
        "\\label{tab:neurolens_vs_baseline}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & Sensitivity (\\%) & FPR/h & AUC-ROC & FP under EMG noise (\\%) \\\\",
        "\\midrule",
        nl_row,
        bl_row,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    latex = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    return latex


def plot_roc_comparison(
    neurolens_probs: np.ndarray,
    neurolens_labels: np.ndarray,
    baseline_probs: np.ndarray,
    baseline_labels: np.ndarray,
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6))
    series = [
        ("NeuroLens", neurolens_probs, neurolens_labels, "#0f7d7d"),
        ("ResNet1D (baseline)", baseline_probs, baseline_labels, "#b8531f"),
    ]
    plotted_any = False
    for label, probs, labels, color in series:
        if probs.size == 0 or len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linewidth=2.2)
        plotted_any = True

    if not plotted_any:
        raise ValueError("Neither model has both classes present in its pooled predictions; cannot draw an ROC curve.")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Comparison: NeuroLens vs. ResNet1D Baseline\n(pooled predictions across LOSO folds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Same data-directory / patient-split arguments as run_experiments.py.
    p.add_argument("--data_dir", type=str, required=True, help="Root of the CHB-MIT database (contains chb01/, chb02/, ...).")
    p.add_argument("--output_dir", type=str, default="runs/experiment", help="Same --output_dir run_experiments.py used.")
    p.add_argument("--patients", type=str, nargs="+", default=["chb01", "chb02", "chb03"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_folds", type=int, default=None, help="Reproduce only the first N NeuroLens folds (smoke tests).")

    # Phase 1: baseline training
    p.add_argument("--epochs_baseline", type=int, default=15)
    p.add_argument("--lr_baseline", type=float, default=1e-3)
    p.add_argument("--baseline_dropout_p", type=float, default=0.3)
    p.add_argument("--lightweight_baseline", action="store_true", help="Use the narrower (base_width=32) ResNet1D variant.")
    p.add_argument("--grad_clip_norm", type=float, default=5.0)

    # Phase 2: baseline evaluation
    p.add_argument("--decision_threshold", type=float, default=0.5)
    p.add_argument("--refractory_sec", type=float, default=300.0)

    # Phase 3: robustness
    p.add_argument("--worst_case_severity", type=float, default=4.0)
    p.add_argument("--robustness_min_windows", type=int, default=None)
    p.add_argument("--robustness_spike_n_std", type=float, default=3.0)
    p.add_argument("--robustness_min_sustain_windows", type=int, default=3)
    p.add_argument("--robustness_mc_samples", type=int, default=20)
    p.add_argument("--robustness_seed", type=int, default=0)
    p.add_argument("--skip_robustness", action="store_true")

    p.add_argument(
        "--neurolens_checkpoint_dir", type=str, default=None,
        help="Defaults to <output_dir>/checkpoints (where run_experiments.py/train.py/evaluate.py wrote NeuroLens's results).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    config = AblationConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        patients=args.patients,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        epochs_baseline=args.epochs_baseline,
        lr_baseline=args.lr_baseline,
        baseline_dropout_p=args.baseline_dropout_p,
        lightweight_baseline=args.lightweight_baseline,
        grad_clip_norm=args.grad_clip_norm,
        decision_threshold=args.decision_threshold,
        refractory_sec=args.refractory_sec,
        worst_case_severity=args.worst_case_severity,
        robustness_min_windows=args.robustness_min_windows,
        robustness_spike_n_std=args.robustness_spike_n_std,
        robustness_min_sustain_windows=args.robustness_min_sustain_windows,
        robustness_mc_samples=args.robustness_mc_samples,
        robustness_seed=args.robustness_seed,
        skip_robustness=args.skip_robustness,
        neurolens_checkpoint_dir=args.neurolens_checkpoint_dir,
        max_folds=args.max_folds,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    baseline_ckpt_dir = os.path.join(config.output_dir, "baseline_checkpoints")
    ablation_dir = os.path.join(config.output_dir, "ablation_artifacts")
    os.makedirs(baseline_ckpt_dir, exist_ok=True)
    os.makedirs(ablation_dir, exist_ok=True)

    log = setup_logging(os.path.join(config.output_dir, "ablation_run.log"))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    log.info(f"NeuroLens ablation study starting. Using device: {device}")
    log.info(f"Config: {json.dumps(asdict(config), indent=2, default=str)}")

    neurolens_checkpoint_dir = config.neurolens_checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
    training_summary_path = os.path.join(neurolens_checkpoint_dir, "training_summary.json")
    neurolens_eval_path = os.path.join(neurolens_checkpoint_dir, "evaluation_report.json")
    if not os.path.isfile(training_summary_path):
        raise FileNotFoundError(
            f"{training_summary_path} not found -- run run_experiments.py (or train.py) for NeuroLens first."
        )
    if not os.path.isfile(neurolens_eval_path):
        raise FileNotFoundError(
            f"{neurolens_eval_path} not found -- run run_experiments.py (or evaluate.py) for NeuroLens first."
        )
    with open(training_summary_path) as f:
        neurolens_folds_meta = json.load(f)["folds"]
    with open(neurolens_eval_path) as f:
        neurolens_eval_report = json.load(f)

    neurolens_folds_meta = [f for f in neurolens_folds_meta if f["patient_id"] in config.patients]
    if config.max_folds is not None:
        neurolens_folds_meta = neurolens_folds_meta[: config.max_folds]
    if not neurolens_folds_meta:
        raise RuntimeError(
            f"No NeuroLens folds found for patients {list(config.patients)} in {training_summary_path}"
        )
    target_fold_ids = {f["fold_id"] for f in neurolens_folds_meta}
    log.info(f"Reproducing {len(neurolens_folds_meta)} NeuroLens fold(s) for the baseline: {sorted(target_fold_ids)}")

    # ---- Phase 1 + 2 -----------------------------------------------------
    baseline_training_summary: List[Dict] = []
    baseline_eval_folds: List[Dict] = []
    fold_model_cache: Dict[str, ResNet1D] = {}
    fold_faiss_k_cache: Dict[str, int] = {}

    for entry in neurolens_folds_meta:
        patient_id, test_seizure_idx, fold_id = entry["patient_id"], entry["test_seizure_idx"], entry["fold_id"]

        log.info(f"=== Phase 1: training baseline for fold {fold_id} ===")
        try:
            result = train_baseline_fold(patient_id, test_seizure_idx, config, device, log)
        except RuntimeError as exc:
            log.warning(f"Skipping baseline fold {fold_id}: {exc}")
            continue

        ckpt_path = os.path.join(baseline_ckpt_dir, f"{fold_id}.pt")
        torch.save(
            {
                "model_state_dict": result.model.state_dict(),
                "model_kwargs": _baseline_model_kwargs(config),
                "val_auprc": result.best_val_auprc,
                "patient_id": patient_id,
                "test_seizure_idx": test_seizure_idx,
                "history": result.history,
            },
            ckpt_path,
        )
        log.info(f"Saved baseline checkpoint: {ckpt_path} (val_auprc={result.best_val_auprc:.4f})")

        log.info(f"=== Phase 2: evaluating baseline for fold {fold_id} ===")
        eval_entry = evaluate_baseline_fold(result.model, patient_id, test_seizure_idx, result.test_loader, config, device, log)
        baseline_eval_folds.append(eval_entry)

        baseline_training_summary.append(
            {
                "fold_id": fold_id,
                "patient_id": patient_id,
                "test_seizure_idx": test_seizure_idx,
                "val_auprc": result.best_val_auprc,
                "checkpoint": ckpt_path,
                "n_train_windows": len(result.train_loader.dataset),
                "n_val_windows": len(result.val_loader.dataset),
                "n_test_windows": len(result.test_loader.dataset),
            }
        )
        fold_model_cache[fold_id] = result.model
        faiss_dir = entry.get("faiss_index_dir")
        k = _read_faiss_k(faiss_dir) if faiss_dir else None
        fold_faiss_k_cache[fold_id] = k if k is not None else 12

    with open(os.path.join(baseline_ckpt_dir, "baseline_training_summary.json"), "w") as f:
        json.dump({"config": asdict(config), "folds": baseline_training_summary}, f, indent=2)

    baseline_eval_report_path = os.path.join(baseline_ckpt_dir, "baseline_evaluation_report.json")
    with open(baseline_eval_report_path, "w") as f:
        json.dump({"folds": baseline_eval_folds}, f, indent=2)
    log.info(f"Phase 1+2 complete: {len(baseline_eval_folds)}/{len(neurolens_folds_meta)} baseline fold(s) trained and evaluated")

    # ---- Phase 3: robustness ---------------------------------------------
    neurolens_robustness_by_fold: Dict[str, Dict] = {}
    baseline_robustness_by_fold: Dict[str, Dict] = {}

    if config.skip_robustness:
        log.info("Phase 3 (robustness) skipped (--skip_robustness)")
    else:
        for entry in neurolens_folds_meta:
            fold_id = entry["fold_id"]
            if fold_id not in fold_model_cache:
                continue  # baseline training failed for this fold; nothing to compare
            patient_id = entry["patient_id"]
            min_windows = config.robustness_min_windows or (fold_faiss_k_cache[fold_id] + 24)

            log.info(f"=== Phase 3: robustness @ severity={config.worst_case_severity:g} for fold {fold_id} ===")
            nl_report = run_neurolens_robustness(patient_id, entry, min_windows, config, log)
            if nl_report is not None:
                neurolens_robustness_by_fold[fold_id] = nl_report

            bl_report = run_baseline_robustness(fold_model_cache[fold_id], patient_id, min_windows, config, device, log)
            if bl_report is not None:
                baseline_robustness_by_fold[fold_id] = bl_report

    with open(os.path.join(ablation_dir, "robustness_comparison.json"), "w") as f:
        json.dump(
            {
                "worst_case_severity": config.worst_case_severity,
                "neurolens": neurolens_robustness_by_fold,
                "baseline": baseline_robustness_by_fold,
            },
            f,
            indent=2,
        )

    # ---- Phase 4: comparative artifacts -----------------------------------
    log.info("=== Phase 4: comparative artifacts ===")
    neurolens_eval_folds = [f for f in neurolens_eval_report.get("folds", []) if f["fold_id"] in target_fold_ids]
    if len(neurolens_eval_folds) != len(neurolens_folds_meta):
        log.warning(
            f"NeuroLens evaluation_report.json has {len(neurolens_eval_folds)} matching fold(s), "
            f"expected {len(neurolens_folds_meta)}; the comparison below covers only the overlap."
        )

    table_path = os.path.join(ablation_dir, "ablation_table.tex")
    latex = build_ablation_table(
        neurolens_eval_folds,
        baseline_eval_folds,
        neurolens_robustness_by_fold,
        baseline_robustness_by_fold,
        config.worst_case_severity,
        table_path,
    )
    log.info(f"Wrote LaTeX ablation table: {table_path}")
    log.info("\n" + latex)

    nl_probs = (
        np.concatenate([np.array(f["probs_calibrated"], dtype=np.float64) for f in neurolens_eval_folds])
        if neurolens_eval_folds
        else np.zeros(0)
    )
    nl_labels = (
        np.concatenate([np.array(f["labels"], dtype=np.int64) for f in neurolens_eval_folds])
        if neurolens_eval_folds
        else np.zeros(0)
    )
    bl_probs = (
        np.concatenate([np.array(f["probs"], dtype=np.float64) for f in baseline_eval_folds])
        if baseline_eval_folds
        else np.zeros(0)
    )
    bl_labels = (
        np.concatenate([np.array(f["labels"], dtype=np.int64) for f in baseline_eval_folds])
        if baseline_eval_folds
        else np.zeros(0)
    )

    plot_path = os.path.join(ablation_dir, "roc_comparison.png")
    try:
        plot_roc_comparison(nl_probs, nl_labels, bl_probs, bl_labels, plot_path)
        log.info(f"Wrote ROC comparison plot: {plot_path}")
    except ValueError as exc:
        log.warning(f"Skipped ROC comparison plot: {exc}")

    log.info("Ablation study complete.")
    log.info(f"  Ablation table: {table_path}")
    log.info(f"  ROC plot:       {plot_path}")
    log.info(f"  Robustness:     {os.path.join(ablation_dir, 'robustness_comparison.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
