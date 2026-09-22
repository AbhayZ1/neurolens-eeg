"""Publication-artifact generator for NeuroLens.

Ingests the JSON reports produced by evaluate.py and (optionally)
run_ablations.py and writes the complete paper package:

    Tables:
        1. results_table.tex       -- Sensitivity, FPR/h, AUC-ROC, AUPRC,
                                       BPE, mean +/- 95% CI, per patient
                                       and pooled.
        2. faithfulness_table.tex  -- Counterfactual Steer Drop (%) and
                                       Wilcoxon significance.
        3. ablation_table.tex      -- NeuroLens vs. ResNet1D baseline
                                       (needs --baseline_report_json and
                                       --robustness_json).

    Figures (all rendered at 300 DPI):
        1. fig1_latent_trajectory_map.png       -- 2D projection (UMAP if
           installed, else PCA) of historical interictal/preictal
           trajectory endpoints, with one fold's own live test-set
           trajectory overlaid as a path from the interictal attractor
           toward the preictal cluster.
        2. fig2_tas_vs_time_to_seizure.png      -- TAS(t) on synthetic
           Jansen-Rit ground truth vs. time to the exact bifurcation
           onset, with the SPH lockout window highlighted.
        3. fig3_roc_pr_comparison.png           -- ROC and Precision-Recall
           curves, NeuroLens vs. ResNet1D baseline (needs
           --baseline_report_json).
        4. fig4_calibration_curves.png          -- Reliability diagram,
           pre- vs. post-temperature-scaling, with ECE/Brier annotated.
        5. fig5_channel_counterfactual_heatmap.png -- Per-bipolar-channel
           bar chart of the mean counterfactual tangent-space delta:
           which physical EEG channels' power/synchrony the steering
           leans on to reduce risk.

This script only needs the JSON reports plus numpy/scipy/matplotlib/
scikit-learn (umap-learn is optional, used for Fig. 1 if present, with an
automatic PCA fallback) -- no torch/mne/faiss -- so it can run on a laptop
after copying just the JSON files off wherever training/evaluation
actually ran.

Every table and figure is generated independently: if the data one needs
is missing (e.g. no --baseline_report_json, or a fold with no exported
FAISS index), that ONE artifact is skipped with a clear printed reason and
every other artifact still gets produced -- this script never aborts the
whole run over one missing piece, and always exits 0.

Usage:
    python generate_paper_artifacts.py \
        --report_json runs/exp1/checkpoints/evaluation_report.json \
        --baseline_report_json runs/exp1/baseline_checkpoints/baseline_evaluation_report.json \
        --robustness_json runs/exp1/ablation_artifacts/robustness_comparison.json \
        --output_dir runs/exp1/paper_artifacts
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display or GUI backend required
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

DPI = 300


# --------------------------------------------------------------------------
# Shared statistics / formatting helpers (self-contained -- no dependency
# on run_experiments.py / run_ablations.py, so this script runs standalone)
# --------------------------------------------------------------------------


def mean_confidence_interval(values: Sequence[Optional[float]], confidence: float = 0.95) -> Dict[str, Optional[float]]:
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


def _counterfactual_steer_drop_pct(fold_report: Dict) -> Optional[float]:
    """% drop in predicted seizure probability after applying delta_z."""
    fa = fold_report.get("faithfulness") or {}
    initial = fa.get("mean_initial_risk")
    final = fa.get("mean_counterfactual_final_risk")
    if initial is None or final is None or initial <= 0:
        return None
    return 100.0 * (initial - final) / initial


def _group_by_patient(folds: List[Dict]) -> Dict[str, List[Dict]]:
    by_patient: Dict[str, List[Dict]] = defaultdict(list)
    for f in folds:
        by_patient[f["patient_id"]].append(f)
    return dict(sorted(by_patient.items()))


def _fmt(stat: Dict, scale: float = 1.0, decimals: int = 3) -> str:
    if stat["mean"] is None:
        return "N/A"
    mean = stat["mean"] * scale
    if stat["half_width"] is None:  # single-fold: no CI is defined
        return f"{mean:.{decimals}f}"
    hw = stat["half_width"] * scale
    return f"{mean:.{decimals}f} $\\pm$ {hw:.{decimals}f}"


def _escape_latex(text: str) -> str:
    return text.replace("_", "\\_")


def _pick_fold(
    folds: List[Dict],
    fold_id: Optional[str],
    require_keys: Sequence[str] = (),
    require_channel_attribution: bool = False,
) -> Dict:
    """Picks `fold_id` if given (and valid), else the first fold in the
    report that actually carries the data the caller needs."""
    if fold_id is not None:
        for f in folds:
            if f["fold_id"] == fold_id:
                candidates = [f]
                break
        else:
            raise ValueError(f"fold_id {fold_id!r} not found in the report")
    else:
        candidates = folds

    for f in candidates:
        has_keys = all(f.get(k) for k in require_keys)
        has_attribution = (not require_channel_attribution) or bool((f.get("faithfulness") or {}).get("channel_attribution"))
        if has_keys and has_attribution:
            return f

    raise ValueError(
        f"No fold with the required data found (require_keys={list(require_keys)}, "
        f"require_channel_attribution={require_channel_attribution}); pass an explicit "
        "--fig1_fold_id/--fig5_fold_id, or re-run evaluate.py with the relevant benchmark enabled."
    )


# --------------------------------------------------------------------------
# Table 1: results_table.tex -- Sensitivity, FPR/h, AUC-ROC, AUPRC, BPE
# --------------------------------------------------------------------------


def build_latex_table(report: Dict, output_path: str) -> str:
    all_folds = report.get("folds", [])
    by_patient = _group_by_patient(all_folds)

    def _row(label: str, fold_list: List[Dict]) -> str:
        sens = mean_confidence_interval([f["clinical"]["sensitivity_window"] for f in fold_list])
        fpr = mean_confidence_interval([f["clinical"]["fpr_per_hour_event"] for f in fold_list])
        auc = mean_confidence_interval([f["clinical"]["auc_roc"] for f in fold_list])
        auprc = mean_confidence_interval([f["clinical"]["auprc"] for f in fold_list])
        bpe = mean_confidence_interval([(f.get("bifurcation") or {}).get("mean_abs_bpe_sec") for f in fold_list])
        return (
            f"{_escape_latex(label)} & {_fmt(sens, scale=100.0, decimals=1)} & {_fmt(fpr, decimals=3)} & "
            f"{_fmt(auc, decimals=3)} & {_fmt(auprc, decimals=3)} & {_fmt(bpe, decimals=2)} \\\\"
        )

    if not all_folds:
        raise ValueError("report['folds'] is empty; nothing to tabulate.")

    lines = [
        "% Auto-generated by generate_paper_artifacts.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Seizure prediction performance across CHB-MIT patients (Leave-One-Seizure-Out "
        "cross-validation), reported as mean $\\pm$ 95\\% CI over folds. FPR/h is the event-level "
        "(refractory-debounced) false prediction rate per hour of interictal recording. BPE is the "
        "Bifurcation Point Error (seconds) on synthetic Jansen-Rit ground-truth data.}",
        "\\label{tab:neurolens_results}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Patient & Sensitivity (\\%) & FPR/h & AUC-ROC & AUPRC & BPE (s) \\\\",
        "\\midrule",
    ]
    for patient_id, fold_list in by_patient.items():
        lines.append(_row(f"{patient_id} (n={len(fold_list)})", fold_list))
    lines.append("\\midrule")
    lines.append(_row(f"Overall (n={len(all_folds)})", all_folds))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    latex = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    return latex


# --------------------------------------------------------------------------
# Table 2: faithfulness_table.tex -- Counterfactual Steer Drop + Wilcoxon
# --------------------------------------------------------------------------


def build_faithfulness_table(report: Dict, output_path: str) -> str:
    all_folds = report.get("folds", [])
    by_patient = _group_by_patient(all_folds)

    def _row(label: str, fold_list: List[Dict]) -> str:
        steer = mean_confidence_interval([_counterfactual_steer_drop_pct(f) for f in fold_list])
        gain = mean_confidence_interval([(f.get("faithfulness") or {}).get("mean_faithfulness_gain") for f in fold_list])
        p_values = [
            (f.get("faithfulness") or {}).get("wilcoxon_p_value")
            for f in fold_list
            if (f.get("faithfulness") or {}).get("wilcoxon_p_value") is not None
        ]
        frac_sig_str = f"{100.0 * sum(1 for p in p_values if p < 0.05) / len(p_values):.0f}\\%" if p_values else "N/A"
        return f"{_escape_latex(label)} & {_fmt(steer, decimals=1)} & {_fmt(gain, decimals=4)} & {frac_sig_str} \\\\"

    if not all_folds:
        raise ValueError("report['folds'] is empty; nothing to tabulate.")

    lines = [
        "% Auto-generated by generate_paper_artifacts.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Counterfactual latent-steering faithfulness: risk reduction from the computed "
        "counterfactual $\\delta z$ versus a magnitude-matched random Gaussian perturbation, mean "
        "$\\pm$ 95\\% CI over LOSO folds. Steer Drop is the percentage drop in MC-Dropout-averaged "
        "predicted seizure probability after applying $\\delta z$; Faithfulness Gain is the risk "
        "reduction advantage of $\\delta z$ over matched-magnitude random noise; the last column is "
        "the fraction of folds where this advantage was significant (paired Wilcoxon $p<0.05$).}",
        "\\label{tab:neurolens_faithfulness}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Patient & Steer Drop (\\%) & Faithfulness Gain & Folds $p<0.05$ \\\\",
        "\\midrule",
    ]
    for patient_id, fold_list in by_patient.items():
        lines.append(_row(f"{patient_id} (n={len(fold_list)})", fold_list))
    lines.append("\\midrule")
    lines.append(_row(f"Overall (n={len(all_folds)})", all_folds))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    latex = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    return latex


# --------------------------------------------------------------------------
# Table 3: ablation_table.tex -- NeuroLens vs. ResNet1D baseline
# --------------------------------------------------------------------------


def _fp_rate_neurolens(robustness_entry: Optional[Dict]) -> Optional[float]:
    """Fraction of windows flagged false-positive at the (single) worst-case
    severity artifact_robustness.py was run at -- the last entry in
    'severities' (index 0 is always the clean/severity=0.0 baseline)."""
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
    if not neurolens_folds or not baseline_folds:
        raise ValueError("Need overlapping NeuroLens and baseline folds to build the ablation table.")

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
        "% Auto-generated by generate_paper_artifacts.py -- do not edit by hand.",
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


# --------------------------------------------------------------------------
# Fig 1: 2D Latent Trajectory Map
# --------------------------------------------------------------------------


def plot_latent_trajectory_map(fold_report: Dict, output_path: str, method: str = "auto") -> str:
    """Historical interictal/preictal trajectory endpoints (from the fold's
    exported FAISS index, read directly as .npy -- no faiss import needed)
    projected to 2D, with this fold's own live test-set trajectory overlaid
    as a time-colored path. Returns which reduction method was actually used.
    """
    faiss_dir = fold_report.get("faiss_index_dir")
    latents = fold_report.get("latent_vectors")
    labels = fold_report.get("labels")
    if not faiss_dir or not os.path.isdir(faiss_dir):
        raise ValueError(f"No FAISS index directory recorded for fold {fold_report.get('fold_id')!r}")
    if not latents:
        raise ValueError(f"No latent_vectors recorded for fold {fold_report.get('fold_id')!r}")

    traj_path = os.path.join(faiss_dir, "trajectories.npy")
    lab_path = os.path.join(faiss_dir, "labels.npy")
    if not os.path.isfile(traj_path) or not os.path.isfile(lab_path):
        raise ValueError(f"{faiss_dir} is missing trajectories.npy/labels.npy")

    hist_trajectories = np.load(traj_path)  # [N, k, d]
    hist_labels = np.load(lab_path)  # [N]
    hist_points = hist_trajectories[:, -1, :]  # each historical trajectory's own "current" endpoint

    live_points = np.asarray(latents, dtype=np.float64)  # [T, d]
    all_points = np.concatenate([hist_points, live_points], axis=0)

    method_used = "PCA"
    coords = None
    if method in ("auto", "umap"):
        try:
            import umap

            coords = umap.UMAP(n_components=2, random_state=42).fit_transform(all_points)
            method_used = "UMAP"
        except ImportError:
            if method == "umap":
                raise ValueError("umap-learn is not installed; install it or pass --fig1_reduction_method pca")
    if coords is None:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=42).fit_transform(all_points)

    n_hist = hist_points.shape[0]
    hist_coords = coords[:n_hist]
    live_coords = coords[n_hist:]

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    for cls, name, color in [(0, "Interictal (historical)", "#3b7dd8"), (1, "Preictal (historical)", "#d84a4a")]:
        mask = hist_labels == cls
        if mask.any():
            ax.scatter(hist_coords[mask, 0], hist_coords[mask, 1], s=10, alpha=0.25, color=color, label=name, linewidths=0)

    n_live = live_coords.shape[0]
    if n_live > 1:
        cmap = plt.get_cmap("plasma")
        for i in range(n_live - 1):
            ax.plot(
                live_coords[i : i + 2, 0], live_coords[i : i + 2, 1],
                color=cmap(i / max(n_live - 2, 1)), linewidth=1.8, alpha=0.9, zorder=5,
            )
    if n_live > 0:
        ax.scatter(live_coords[:, 0], live_coords[:, 1], c=np.arange(n_live), cmap="plasma", s=22, zorder=6, edgecolors="black", linewidths=0.3)
        ax.scatter(live_coords[0, 0], live_coords[0, 1], s=150, marker="o", facecolors="none", edgecolors="black", linewidths=2, zorder=7, label="Live stream: start")
        ax.scatter(live_coords[-1, 0], live_coords[-1, 1], s=240, marker="*", color="gold", edgecolors="black", linewidths=1, zorder=7, label="Live stream: end (at SPH boundary)")

    ax.set_xlabel(f"{method_used} dimension 1")
    ax.set_ylabel(f"{method_used} dimension 2")
    ax.set_title(
        f"2D Latent Trajectory Map -- {fold_report.get('fold_id', '')}\n"
        "Live stream: interictal attractor $\\rightarrow$ preictal cluster",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    return method_used


# --------------------------------------------------------------------------
# Fig 2: TAS vs. Time-to-Seizure
# --------------------------------------------------------------------------


def plot_tas_vs_time_to_seizure(report: Dict, output_path: str, max_curves: Optional[int] = None) -> None:
    entries = []
    for f in report.get("folds", []):
        bif = f.get("bifurcation")
        if not bif:
            continue
        sph_sec = bif.get("sph_sec")
        for trial in bif.get("trials", []):
            tas_times = trial.get("tas_times_sec")
            tas_series = trial.get("tas_series")
            true_onset = trial.get("true_bifurcation_onset_sec")
            if not tas_times or not tas_series or true_onset is None:
                continue
            entries.append(
                {
                    "fold_id": f["fold_id"],
                    "trial": trial["trial"],
                    "t": np.asarray(tas_times, dtype=np.float64) - true_onset,
                    "y": np.asarray(tas_series, dtype=np.float64),
                    "detected_onset_sec": trial.get("detected_onset_sec"),
                    "true_onset": true_onset,
                    "sph_sec": sph_sec,
                }
            )

    if not entries:
        raise ValueError(
            "No TAS(t) curves found in this report's bifurcation results. Either "
            "--no_bifurcation_benchmark was passed to evaluate.py, or every fold hit the "
            "'no historical preictal trajectories in this fold's FAISS index' fallback "
            "(check for that note in evaluation_report.json's folds[*].bifurcation.trials)."
        )

    if max_curves is not None:
        entries = entries[:max_curves]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    sph_sec = next((e["sph_sec"] for e in entries if e["sph_sec"] is not None), 300.0)

    for i, e in enumerate(entries):
        color = cmap(i % 10)
        ax.plot(e["t"], e["y"], color=color, alpha=0.85, linewidth=1.5, label=f"{e['fold_id']} (trial {e['trial']})")
        if e["detected_onset_sec"] is not None:
            t_detect = e["detected_onset_sec"] - e["true_onset"]
            ax.axvline(t_detect, color=color, linestyle=":", linewidth=1.2, alpha=0.8)

    ax.axvline(0.0, color="black", linestyle="-", linewidth=1.5, label="Bifurcation onset ($t=0$)")
    ax.axvline(-sph_sec, color="crimson", linestyle="--", linewidth=1.5, label=f"SPH boundary ($-{sph_sec / 60:.0f}$ min)")
    ax.axvspan(-sph_sec, 0.0, color="crimson", alpha=0.08, label="Excluded SPH window")

    ax.set_xlabel("Time to Seizure / Bifurcation Onset (s)")
    ax.set_ylabel("Trajectory Alignment Score (TAS) vs. historical preictal trajectories")
    ax.set_title("TAS(t) on Synthetic Jansen-Rit Ground Truth vs. Time to Seizure")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 3: ROC & Precision-Recall Comparison
# --------------------------------------------------------------------------


def plot_roc_pr_comparison(
    neurolens_probs: np.ndarray,
    neurolens_labels: np.ndarray,
    baseline_probs: np.ndarray,
    baseline_labels: np.ndarray,
    output_path: str,
) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5.5))
    series = [
        ("NeuroLens", neurolens_probs, neurolens_labels, "#0f7d7d"),
        ("ResNet1D (baseline)", baseline_probs, baseline_labels, "#b8531f"),
    ]

    plotted_any = False
    base_rate = None
    for label, probs, labels, color in series:
        if probs.size == 0 or len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax_roc.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linewidth=2.2)

        precision, recall, _ = precision_recall_curve(labels, probs)
        auprc = average_precision_score(labels, probs)
        ax_pr.plot(recall, precision, label=f"{label} (AUPRC={auprc:.3f})", color=color, linewidth=2.2)
        if base_rate is None:
            base_rate = float(np.mean(labels))
        plotted_any = True

    if not plotted_any:
        raise ValueError("Neither model has both classes present in its pooled predictions; cannot draw ROC/PR curves.")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.02)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(alpha=0.3)

    if base_rate is not None:
        ax_pr.axhline(base_rate, linestyle="--", color="gray", linewidth=1, label=f"Chance (base rate={base_rate:.3f})")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.02)
    ax_pr.set_xlabel("Recall (Sensitivity)")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.grid(alpha=0.3)

    fig.suptitle("NeuroLens vs. ResNet1D Baseline (pooled predictions across LOSO folds)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 4: Calibration / Reliability Curve, pre- vs. post-temperature-scaling
# --------------------------------------------------------------------------


def plot_calibration_curves(report: Dict, output_path: str) -> None:
    agg = report.get("aggregate", {})
    unc_raw = agg.get("pooled_uncertainty_raw") or {}
    unc_cal = agg.get("pooled_uncertainty") or {}
    curve_raw = unc_raw.get("calibration_curve")
    curve_cal = unc_cal.get("calibration_curve")
    if not curve_raw or not curve_cal:
        raise ValueError(
            "Pooled calibration curves not found in report['aggregate']; re-run evaluate.py with the "
            "current schema (needs both 'pooled_uncertainty_raw' and 'pooled_uncertainty')."
        )

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")

    any_plotted = False
    for curve, label, color, ece, brier in [
        (curve_raw, "Pre-temperature-scaling", "#c0392b", unc_raw.get("ece"), unc_raw.get("brier_score")),
        (curve_cal, "Post-temperature-scaling", "#1f7a4d", unc_cal.get("ece"), unc_cal.get("brier_score")),
    ]:
        conf = [b["confidence"] for b in curve if b["count"] > 0]
        acc = [b["accuracy"] for b in curve if b["count"] > 0]
        if not conf:
            continue
        ece_str = f"{ece:.3f}" if ece is not None else "N/A"
        brier_str = f"{brier:.3f}" if brier is not None else "N/A"
        ax.plot(conf, acc, color=color, linewidth=1.8, marker="o", markersize=5, label=f"{label} (ECE={ece_str}, Brier={brier_str})")
        any_plotted = True

    if not any_plotted:
        raise ValueError("Both pooled calibration curves are empty (no non-empty probability bins).")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (confidence)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram: Pre- vs. Post-Temperature-Scaling\n(pooled across LOSO folds)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 5: Channel-Level Counterfactual Heatmap
# --------------------------------------------------------------------------


def plot_channel_counterfactual_heatmap(fold_report: Dict, output_path: str, top_n: Optional[int] = None) -> None:
    attribution = (fold_report.get("faithfulness") or {}).get("channel_attribution")
    if not attribution:
        raise ValueError(f"No channel_attribution recorded for fold {fold_report.get('fold_id')!r}")

    channel_power_delta = attribution["channel_power_delta"]
    channels = list(channel_power_delta.keys())
    values = np.array([channel_power_delta[c] for c in channels], dtype=np.float64)

    if top_n is not None and top_n < len(channels):
        keep = np.argsort(-np.abs(values))[:top_n]
        order = keep[np.argsort(values[keep])]
    else:
        order = np.argsort(values)

    sorted_channels = [channels[i] for i in order]
    sorted_values = values[order]
    colors = ["#c0392b" if v > 0 else "#1f6fb2" for v in sorted_values]

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.35 * len(sorted_channels) + 1.5)))
    ax.barh(sorted_channels, sorted_values, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("$\\Delta$ log-covariance (power / synchrony) from counterfactual steering")
    ax.set_title(
        f"Channel-Level Counterfactual Attribution -- {fold_report.get('fold_id', '')}\n"
        f"Dominant: {attribution['dominant_region']} ({attribution['dominant_direction']})"
    )
    ax.legend(
        handles=[
            mpatches.Patch(color="#c0392b", label="Increasing this channel reduces risk"),
            mpatches.Patch(color="#1f6fb2", label="Decreasing this channel reduces risk"),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Orchestration: run every artifact independently, never let one failure
# take down the rest (this script is meant to run unattended at the end of
# a multi-hour Kaggle job).
# --------------------------------------------------------------------------


def _run_step(name: str, fn, results: Dict[str, str]) -> None:
    try:
        fn()
        results[name] = "OK"
        print(f"[OK]      {name}")
    except Exception as exc:
        results[name] = f"SKIPPED: {exc}"
        print(f"[SKIPPED] {name}: {exc}")


def _pooled_probs_labels(folds: List[Dict], probs_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if not folds:
        return np.zeros(0), np.zeros(0)
    probs = np.concatenate([np.array(f[probs_key], dtype=np.float64) for f in folds])
    labels = np.concatenate([np.array(f["labels"], dtype=np.int64) for f in folds])
    return probs, labels


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report_json", type=str, required=True, help="Path to evaluate.py's evaluation_report.json (NeuroLens).")
    p.add_argument(
        "--baseline_report_json", type=str, default=None,
        help="Path to run_ablations.py's baseline_evaluation_report.json. Enables ablation_table.tex and Fig. 3.",
    )
    p.add_argument(
        "--robustness_json", type=str, default=None,
        help="Path to run_ablations.py's robustness_comparison.json. Enables ablation_table.tex's FP-under-EMG-noise column.",
    )
    p.add_argument("--output_dir", type=str, default="artifacts")

    p.add_argument("--table_filename", type=str, default="results_table.tex")
    p.add_argument("--faithfulness_table_filename", type=str, default="faithfulness_table.tex")
    p.add_argument("--ablation_table_filename", type=str, default="ablation_table.tex")
    p.add_argument("--fig1_filename", type=str, default="fig1_latent_trajectory_map.png")
    p.add_argument("--fig2_filename", type=str, default="fig2_tas_vs_time_to_seizure.png")
    p.add_argument("--fig3_filename", type=str, default="fig3_roc_pr_comparison.png")
    p.add_argument("--fig4_filename", type=str, default="fig4_calibration_curves.png")
    p.add_argument("--fig5_filename", type=str, default="fig5_channel_counterfactual_heatmap.png")

    p.add_argument("--fig1_fold_id", type=str, default=None, help="Which fold's live trajectory to overlay in Fig. 1; defaults to the first fold with the needed data.")
    p.add_argument("--fig1_reduction_method", type=str, default="auto", choices=["auto", "umap", "pca"])
    p.add_argument("--fig5_fold_id", type=str, default=None, help="Which fold's channel attribution to plot in Fig. 5; defaults to the first fold with the needed data.")
    p.add_argument("--fig5_top_n", type=int, default=None, help="Only plot the top-N channels by |delta| in Fig. 5 (default: all 18).")
    p.add_argument("--max_curves_plotted", type=int, default=None, help="Cap the number of fold/trial curves drawn on Fig. 2.")
    p.add_argument("--worst_case_severity", type=float, default=None, help="Overrides the severity label on ablation_table.tex; defaults to the value recorded in --robustness_json.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.report_json) as f:
        report = json.load(f)
    neurolens_folds = report.get("folds", [])

    baseline_report = None
    if args.baseline_report_json:
        with open(args.baseline_report_json) as f:
            baseline_report = json.load(f)

    robustness = None
    if args.robustness_json:
        with open(args.robustness_json) as f:
            robustness = json.load(f)

    results: Dict[str, str] = {}

    _run_step(
        "results_table.tex",
        lambda: build_latex_table(report, os.path.join(args.output_dir, args.table_filename)),
        results,
    )
    _run_step(
        "faithfulness_table.tex",
        lambda: build_faithfulness_table(report, os.path.join(args.output_dir, args.faithfulness_table_filename)),
        results,
    )

    if baseline_report is not None and robustness is not None:
        worst_case_severity = args.worst_case_severity if args.worst_case_severity is not None else robustness.get("worst_case_severity", 4.0)
        overlap = {f["fold_id"] for f in neurolens_folds} & {f["fold_id"] for f in baseline_report.get("folds", [])}
        nl_overlap = [f for f in neurolens_folds if f["fold_id"] in overlap]
        bl_overlap = [f for f in baseline_report["folds"] if f["fold_id"] in overlap]
        _run_step(
            "ablation_table.tex",
            lambda: build_ablation_table(
                nl_overlap, bl_overlap, robustness.get("neurolens", {}), robustness.get("baseline", {}),
                worst_case_severity, os.path.join(args.output_dir, args.ablation_table_filename),
            ),
            results,
        )
    else:
        results["ablation_table.tex"] = "SKIPPED: --baseline_report_json and/or --robustness_json not provided"
        print(f"[SKIPPED] ablation_table.tex: {results['ablation_table.tex'][9:]}")

    def _fig1():
        fold = _pick_fold(neurolens_folds, args.fig1_fold_id, require_keys=("faiss_index_dir", "latent_vectors"))
        method = plot_latent_trajectory_map(fold, os.path.join(args.output_dir, args.fig1_filename), method=args.fig1_reduction_method)
        print(f"          (fold={fold['fold_id']}, reduction={method})")

    _run_step("fig1_latent_trajectory_map.png", _fig1, results)

    _run_step(
        "fig2_tas_vs_time_to_seizure.png",
        lambda: plot_tas_vs_time_to_seizure(report, os.path.join(args.output_dir, args.fig2_filename), args.max_curves_plotted),
        results,
    )

    def _fig3():
        if baseline_report is None:
            raise ValueError("--baseline_report_json not provided")
        nl_probs, nl_labels = _pooled_probs_labels(neurolens_folds, "probs_calibrated")
        bl_probs, bl_labels = _pooled_probs_labels(baseline_report.get("folds", []), "probs")
        plot_roc_pr_comparison(nl_probs, nl_labels, bl_probs, bl_labels, os.path.join(args.output_dir, args.fig3_filename))

    _run_step("fig3_roc_pr_comparison.png", _fig3, results)

    _run_step(
        "fig4_calibration_curves.png",
        lambda: plot_calibration_curves(report, os.path.join(args.output_dir, args.fig4_filename)),
        results,
    )

    def _fig5():
        fold = _pick_fold(neurolens_folds, args.fig5_fold_id, require_channel_attribution=True)
        plot_channel_counterfactual_heatmap(fold, os.path.join(args.output_dir, args.fig5_filename), top_n=args.fig5_top_n)
        print(f"          (fold={fold['fold_id']})")

    _run_step("fig5_channel_counterfactual_heatmap.png", _fig5, results)

    print("\n=== generate_paper_artifacts.py summary ===")
    n_ok = 0
    for name, status in results.items():
        print(f"  {name}: {status}")
        n_ok += status == "OK"
    print(f"\n{n_ok}/{len(results)} artifacts generated in {args.output_dir}")

    with open(os.path.join(args.output_dir, "artifact_generation_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    return 0  # zero-fail: a partial artifact set is still a successful run of this script


if __name__ == "__main__":
    raise SystemExit(main())
