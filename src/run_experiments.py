"""Master orchestrator for the NeuroLens experimental pipeline.

Runs, in order, logging everything (system info, subprocess output,
per-fold metrics, memory checkpoints) to `<output_dir>/experiment_run.log`:

    Stage 0 -- Synthetic Jansen-Rit ground-truth generation & sanity check.
               Runs in-process (synthetic_nmm.py is a library with no CLI
               and no dependency on a trained model); generates the same
               synthetic recording evaluate.py will later re-derive its own
               trials from, validates it is finite, logs the analytically-
               exact bifurcation point, and archives it to disk. This is a
               fail-fast check: if the JR-NMM integrator is broken, you find
               out in seconds, not after an hours-long training run.

    Stage 1 -- Training, via `train.py` (run as an isolated subprocess, so
               a CUDA OOM or other crash there can't take this orchestrator
               down with it). For every (patient, held-out seizure) LOSO
               fold, train.py's train_one_fold() runs, in order: Phase A
               (contrastive pretraining with DynamicalSupConLoss), Phase B
               (linear probing / fine-tuning with the classification head
               and MC Dropout), post-hoc temperature-scaling calibration,
               and then build_faiss_index_for_fold() exports that fold's
               historical-trajectory FAISS index. train.py's own per-epoch
               log lines are relayed here with a "[train]" prefix, so all
               four sub-steps are visible in this single log even though
               they're one subprocess invocation -- train.py doesn't expose
               them as separately-invokable stages, since Phase A/B share
               one live model instance per fold.

    Stage 2 -- Evaluation, via `evaluate.py` (subprocess): MC-Dropout
               clinical metrics, calibration, counterfactual faithfulness,
               and the synthetic Bifurcation Point Error benchmark, on each
               fold's held-out continuous chronological test span.

    Stage 3 -- Statistical aggregation: reads evaluate.py's
               evaluation_report.json and computes mean +/- 95% CI (a
               t-distribution critical value, appropriate for the small
               number of LOSO folds typical here -- not a large-sample
               normal approximation) for every headline metric, across
               folds, written to `<output_dir>/statistical_summary.json`.

Usage:
    python run_experiments.py --data_dir /path/to/chb-mit-scalp-eeg-database-1.0.0 \
        --output_dir runs/exp1 --patients chb01 chb02 chb03 --device cuda

Then feed the resulting evaluation_report.json to generate_paper_artifacts.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy import stats

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when psutil is absent
    psutil = None
    _PSUTIL_AVAILABLE = False

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)  # allow `python path/to/run_experiments.py` from anywhere

TRAIN_SCRIPT = os.path.join(SRC_DIR, "train.py")
EVALUATE_SCRIPT = os.path.join(SRC_DIR, "evaluate.py")

from synthetic_nmm import JansenRitParams, generate_synthetic_bifurcation_dataset  # noqa: E402


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("neurolens.orchestrator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def log_system_info(log: logging.Logger) -> None:
    log.info("=" * 78)
    log.info("SYSTEM / ENVIRONMENT")
    log.info("=" * 78)
    log.info(f"Python: {platform.python_version()} ({platform.platform()})")
    log.info(f"PyTorch: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            try:
                free_b, total_b = torch.cuda.mem_get_info(i)
                mem_str = f"{free_b / 1e9:.2f} GB free / {total_b / 1e9:.2f} GB total"
            except RuntimeError:
                mem_str = f"{props.total_memory / 1e9:.2f} GB total"
            log.info(f"  GPU {i}: {props.name}, {mem_str}, compute capability {props.major}.{props.minor}")
    else:
        log.info("  No CUDA device visible; running on CPU (expect much longer training/eval times).")
    log.info(f"CPU logical cores: {os.cpu_count()}")
    if _PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        log.info(f"System RAM: {vm.total / 1e9:.2f} GB total, {vm.available / 1e9:.2f} GB available")
        log.info(f"Orchestrator process RSS at startup: {psutil.Process().memory_info().rss / 1e6:.1f} MB")
    else:
        log.info("psutil not installed; skipping detailed RAM reporting (pip install psutil to enable).")
    log.info("=" * 78)


def log_memory_checkpoint(log: logging.Logger, label: str) -> None:
    if _PSUTIL_AVAILABLE:
        rss = psutil.Process().memory_info().rss / 1e6
        log.info(f"[memory] {label}: orchestrator process RSS = {rss:.1f} MB")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e6
        peak = torch.cuda.max_memory_allocated() / 1e6
        log.info(f"[memory] {label}: CUDA allocated = {alloc:.1f} MB (peak {peak:.1f} MB) in the orchestrator's own process")
        log.info("[memory] note: train.py/evaluate.py run as separate processes with independent CUDA allocations; see their own [train]/[evaluate] log lines above for their memory footprint.")


# --------------------------------------------------------------------------
# Stage 0: synthetic ground-truth sanity check
# --------------------------------------------------------------------------


def run_stage0_synthetic_sanity_check(args: argparse.Namespace, log: logging.Logger, output_dir: str) -> Dict:
    log.info("=" * 78)
    log.info("STAGE 0: Synthetic Jansen-Rit ground-truth generation & sanity check")
    log.info("=" * 78)

    t0 = time.time()
    params = JansenRitParams()
    synth = generate_synthetic_bifurcation_dataset(
        duration_mins=args.synthetic_duration_min,
        sampling_rate=256,
        n_channels=18,
        transition_start_min=args.synthetic_transition_start_min,
        transition_duration_min=args.synthetic_transition_duration_min,
        params=params,
        seed=args.synthetic_seed0,
    )
    elapsed = time.time() - t0
    gt = synth.ground_truth

    eeg_np = synth.eeg_uv.numpy()
    log.info(f"Generated {args.synthetic_duration_min:.1f} min synthetic EEG in {elapsed:.1f}s")
    log.info(f"  eeg_uv shape: {tuple(eeg_np.shape)} (channels, samples), dtype={eeg_np.dtype}")
    log.info(f"  sampling_rate: {synth.sampling_rate} Hz, n_channels: {len(synth.channel_names)}")
    log.info(
        f"  Hopf bifurcation (numerically located, not hard-coded): "
        f"p_crit={gt.p_crit:.4f} pulses/s, p_pre={gt.p_pre:.4f}, p_post={gt.p_post:.4f}"
    )
    log.info(f"  True bifurcation onset: {gt.bifurcation_onset_sec:.4f}s (sample {gt.bifurcation_onset_sample})")
    log.info(f"  Regime fully established by: {gt.regime_established_sec:.4f}s")

    if not np.isfinite(eeg_np).all():
        raise RuntimeError(
            "Synthetic EEG contains non-finite values (NaN/Inf) -- aborting before an "
            "expensive training run. Check synthetic_nmm.py / JansenRitParams."
        )
    log.info("  Sanity check passed: synthetic EEG is fully finite.")

    npz_path = os.path.join(output_dir, "synthetic_ground_truth.npz")
    np.savez(
        npz_path,
        eeg_uv=eeg_np,
        times_sec=synth.times_sec.numpy(),
        p_drive=synth.p_drive.numpy(),
        stability_index=synth.stability_index.numpy(),
        bifurcation_onset_sec=gt.bifurcation_onset_sec,
        p_crit=gt.p_crit,
    )
    log.info(f"  Archived reference synthetic recording: {npz_path}")
    log_memory_checkpoint(log, "after Stage 0")

    return {
        "generation_seconds": elapsed,
        "eeg_shape": list(eeg_np.shape),
        "p_crit": gt.p_crit,
        "bifurcation_onset_sec": gt.bifurcation_onset_sec,
        "npz_path": npz_path,
    }


# --------------------------------------------------------------------------
# Subprocess streaming
# --------------------------------------------------------------------------


def _stream_subprocess(cmd: List[str], log: logging.Logger, stage_name: str) -> float:
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
    if process.returncode != 0:
        raise RuntimeError(f"[{stage_name}] subprocess exited with code {process.returncode} after {elapsed:.1f}s")
    log.info(f"[{stage_name}] completed successfully in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return elapsed


# --------------------------------------------------------------------------
# Stage 1: training (Phase A + Phase B + FAISS export, per LOSO fold)
# --------------------------------------------------------------------------


def run_stage1_training(args: argparse.Namespace, log: logging.Logger) -> str:
    log.info("=" * 78)
    log.info(
        "STAGE 1: Training -- per LOSO fold: Phase A (contrastive pretrain) -> "
        "Phase B (linear-probe/fine-tune) -> temperature scaling -> FAISS export"
    )
    log.info("=" * 78)

    cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--data_dir", args.data_dir,
        "--output_dir", args.checkpoint_dir,
        "--patients", *args.patients,
        "--device", args.device,
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--val_fraction", str(args.val_fraction),
        "--num_workers", str(args.num_workers),
        "--epochs_pretrain", str(args.epochs_pretrain),
        "--lr_pretrain", str(args.lr_pretrain),
        "--epochs_finetune", str(args.epochs_finetune),
        "--lr_finetune", str(args.lr_finetune),
        "--d_model", str(args.d_model),
        "--n_heads", str(args.n_heads),
        "--n_layers", str(args.n_layers),
        "--d_ff", str(args.d_ff),
        "--dropout_p", str(args.dropout_p),
        "--latent_dim", str(args.latent_dim),
        "--trajectory_k", str(args.trajectory_k),
        "--trajectory_stride", str(args.trajectory_stride),
    ]
    if args.no_freeze_encoder:
        cmd.append("--no_freeze_encoder")
    if args.max_folds_per_patient is not None:
        cmd += ["--max_folds_per_patient", str(args.max_folds_per_patient)]
    if args.extra_train_args:
        cmd += shlex.split(args.extra_train_args)

    _stream_subprocess(cmd, log, "train")
    log_memory_checkpoint(log, "after Stage 1 (training)")

    summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
    if not os.path.isfile(summary_path):
        raise RuntimeError(f"train.py reported success but {summary_path} was not produced")
    with open(summary_path) as f:
        summary = json.load(f)
    n_folds = len(summary.get("folds", []))
    log.info(f"Stage 1 produced {n_folds} trained fold checkpoint(s) in {args.checkpoint_dir}")
    if n_folds == 0:
        raise RuntimeError("No folds were successfully trained; aborting before evaluation.")
    return args.checkpoint_dir


# --------------------------------------------------------------------------
# Stage 2: evaluation
# --------------------------------------------------------------------------


def run_stage2_evaluation(args: argparse.Namespace, log: logging.Logger, checkpoint_dir: str) -> str:
    log.info("=" * 78)
    log.info(
        "STAGE 2: Evaluation -- clinical + uncertainty + XAI-faithfulness + "
        "synthetic bifurcation benchmark on each fold's held-out chronological test span"
    )
    log.info("=" * 78)

    output_json = os.path.join(checkpoint_dir, "evaluation_report.json")
    cmd = [
        sys.executable,
        EVALUATE_SCRIPT,
        "--data_dir", args.data_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--output_json", output_json,
        "--device", args.device,
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--mc_samples", str(args.mc_samples),
        "--decision_threshold", str(args.decision_threshold),
        "--refractory_sec", str(args.refractory_sec),
        "--faithfulness_n_samples", str(args.faithfulness_n_samples),
        "--faithfulness_target_risk", str(args.faithfulness_target_risk),
        "--synthetic_duration_min", str(args.synthetic_duration_min),
        "--synthetic_transition_start_min", str(args.synthetic_transition_start_min),
        "--synthetic_transition_duration_min", str(args.synthetic_transition_duration_min),
        "--synthetic_n_trials", str(args.synthetic_n_trials),
    ]
    if args.no_bifurcation_benchmark:
        cmd.append("--no_bifurcation_benchmark")
    if args.max_folds is not None:
        cmd += ["--max_folds", str(args.max_folds)]
    if args.extra_eval_args:
        cmd += shlex.split(args.extra_eval_args)

    _stream_subprocess(cmd, log, "evaluate")
    log_memory_checkpoint(log, "after Stage 2 (evaluation)")

    if not os.path.isfile(output_json):
        raise RuntimeError(f"evaluate.py reported success but {output_json} was not produced")
    log.info(f"Stage 2 wrote evaluation report: {output_json}")
    return output_json


# --------------------------------------------------------------------------
# Stage 3: statistical aggregation (mean +/- 95% CI across LOSO folds)
# --------------------------------------------------------------------------


def mean_confidence_interval(values: Sequence[Optional[float]], confidence: float = 0.95) -> Dict[str, Optional[float]]:
    """Mean and a Student's-t-based CI -- appropriate here because the
    number of LOSO folds is typically small (a handful to a few dozen),
    where the large-sample normal approximation underestimates uncertainty.
    """
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    n = arr.size
    if n == 0:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "n": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci_low": None, "ci_high": None, "n": 1}
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    half_width = t_crit * sem
    return {"mean": mean, "std": std, "ci_low": mean - half_width, "ci_high": mean + half_width, "n": n}


def _counterfactual_steer_drop_pct(fold_report: Dict) -> Optional[float]:
    """% drop in predicted seizure probability after applying delta_z:
    100 * (initial_risk - counterfactual_final_risk) / initial_risk."""
    fa = fold_report.get("faithfulness") or {}
    initial = fa.get("mean_initial_risk")
    final = fa.get("mean_counterfactual_final_risk")
    if initial is None or final is None or initial <= 0:
        return None
    return 100.0 * (initial - final) / initial


_METRIC_EXTRACTORS = {
    "sensitivity_window": lambda f: f["clinical"]["sensitivity_window"],
    "specificity": lambda f: f["clinical"]["specificity"],
    "fpr_per_hour_window": lambda f: f["clinical"]["fpr_per_hour_window"],
    "fpr_per_hour_event": lambda f: f["clinical"]["fpr_per_hour_event"],
    "auc_roc": lambda f: f["clinical"]["auc_roc"],
    "auprc": lambda f: f["clinical"]["auprc"],
    "brier_score_calibrated": lambda f: f["uncertainty_calibrated"]["brier_score"],
    "ece_calibrated": lambda f: f["uncertainty_calibrated"]["ece"],
    "brier_score_raw": lambda f: f["uncertainty_raw"]["brier_score"],
    "ece_raw": lambda f: f["uncertainty_raw"]["ece"],
    "faithfulness_gain": lambda f: (f.get("faithfulness") or {}).get("mean_faithfulness_gain"),
    "counterfactual_steer_drop_pct": _counterfactual_steer_drop_pct,
    "bifurcation_abs_bpe_sec": lambda f: (f.get("bifurcation") or {}).get("mean_abs_bpe_sec"),
}


def run_stage3_statistics(report_path: str, output_dir: str, log: logging.Logger) -> Dict:
    log.info("=" * 78)
    log.info("STAGE 3: Statistical aggregation across LOSO folds (mean +/- 95% CI)")
    log.info("=" * 78)
    with open(report_path) as f:
        report = json.load(f)
    folds = report.get("folds", [])
    if not folds:
        raise RuntimeError(f"{report_path} contains no evaluated folds")

    summary: Dict = {"n_folds": len(folds), "metrics": {}}
    header = f"{'metric':32s} {'mean':>10s} {'95% CI':>24s} {'n':>4s}"
    log.info(header)
    log.info("-" * len(header))
    for name, extractor in _METRIC_EXTRACTORS.items():
        values = []
        for f in folds:
            try:
                values.append(extractor(f))
            except (KeyError, TypeError):
                values.append(None)
        stat = mean_confidence_interval(values)
        summary["metrics"][name] = stat
        if stat["mean"] is None:
            log.info(f"{name:32s} {'N/A':>10s} {'':>24s} {stat['n']:>4d}")
        elif stat["ci_low"] is None:
            log.info(f"{name:32s} {stat['mean']:>10.4f} {'(n=1, no CI)':>24s} {stat['n']:>4d}")
        else:
            ci_str = f"[{stat['ci_low']:.4f}, {stat['ci_high']:.4f}]"
            log.info(f"{name:32s} {stat['mean']:>10.4f} {ci_str:>24s} {stat['n']:>4d}")

    # evaluate.py's own pooled/event-level numbers are aggregate-only (not
    # meaningfully expressible as a per-fold CI); surface them here too.
    summary["evaluate_py_aggregate"] = report.get("aggregate", {})

    out_path = os.path.join(output_dir, "statistical_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Statistical summary written to {out_path}")
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, required=True, help="Root of the CHB-MIT database (contains chb01/, chb02/, ...).")
    p.add_argument("--output_dir", type=str, default="runs/experiment")
    p.add_argument("--patients", type=str, nargs="+", default=["chb01", "chb02", "chb03"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    # forwarded to train.py
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs_pretrain", type=int, default=10)
    p.add_argument("--lr_pretrain", type=float, default=1e-3)
    p.add_argument("--epochs_finetune", type=int, default=15)
    p.add_argument("--lr_finetune", type=float, default=1e-4)
    p.add_argument("--no_freeze_encoder", action="store_true", help="Fine-tune the whole network in Phase B instead of linear-probing.")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--dropout_p", type=float, default=0.3)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--trajectory_k", type=int, default=12)
    p.add_argument("--trajectory_stride", type=int, default=1)
    p.add_argument("--max_folds_per_patient", type=int, default=None)
    p.add_argument("--extra_train_args", type=str, default=None, help="Extra raw CLI args forwarded verbatim to train.py, e.g. '--grad_clip_norm 1.0'.")

    # forwarded to evaluate.py
    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--decision_threshold", type=float, default=0.5)
    p.add_argument("--refractory_sec", type=float, default=300.0)
    p.add_argument("--faithfulness_n_samples", type=int, default=20)
    p.add_argument("--faithfulness_target_risk", type=float, default=0.1)
    p.add_argument("--no_bifurcation_benchmark", action="store_true")
    p.add_argument("--max_folds", type=int, default=None, help="Evaluate only the first N folds (smoke tests).")
    p.add_argument("--extra_eval_args", type=str, default=None, help="Extra raw CLI args forwarded verbatim to evaluate.py.")

    # shared by Stage 0's sanity check and Stage 2's bifurcation benchmark
    p.add_argument("--synthetic_duration_min", type=float, default=15.0)
    p.add_argument("--synthetic_transition_start_min", type=float, default=10.0)
    p.add_argument("--synthetic_transition_duration_min", type=float, default=0.5)
    p.add_argument("--synthetic_n_trials", type=int, default=2)
    p.add_argument("--synthetic_seed0", type=int, default=1000)

    p.add_argument("--skip_training", action="store_true", help="Skip Stage 1 and evaluate an existing <output_dir>/checkpoints.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "experiment_run.log")
    log = setup_logging(log_path)

    log.info("NeuroLens experimental pipeline starting")
    log.info(f"Command line: {' '.join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))}")
    log.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    log_system_info(log)

    run_config_path = os.path.join(args.output_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info(f"Saved run configuration: {run_config_path}")

    pipeline_t0 = time.time()
    report_path: Optional[str] = None
    try:
        stage0_result = run_stage0_synthetic_sanity_check(args, log, args.output_dir)

        if not args.skip_training:
            run_stage1_training(args, log)
        else:
            log.info(f"Stage 1 skipped (--skip_training); reusing existing checkpoints in {args.checkpoint_dir}")
            if not os.path.isfile(os.path.join(args.checkpoint_dir, "training_summary.json")):
                raise RuntimeError("--skip_training given but no training_summary.json found in checkpoint_dir")

        report_path = run_stage2_evaluation(args, log, args.checkpoint_dir)
        stats_summary = run_stage3_statistics(report_path, args.output_dir, log)

        with open(os.path.join(args.output_dir, "experiment_summary.json"), "w") as f:
            json.dump(
                {
                    "stage0_synthetic_sanity_check": stage0_result,
                    "evaluation_report": report_path,
                    "statistical_summary": stats_summary,
                    "total_seconds": time.time() - pipeline_t0,
                },
                f,
                indent=2,
            )
    except Exception:
        log.exception("Pipeline FAILED")
        log.info(f"Total elapsed before failure: {(time.time() - pipeline_t0) / 60:.1f} min")
        return 1

    total_elapsed = time.time() - pipeline_t0
    log.info("=" * 78)
    log.info(f"PIPELINE COMPLETE in {total_elapsed / 60:.1f} min")
    log.info(f"  Evaluation report:   {report_path}")
    log.info(f"  Statistical summary: {os.path.join(args.output_dir, 'statistical_summary.json')}")
    log.info(f"  Log file:            {log_path}")
    log.info(
        "Next: python generate_paper_artifacts.py "
        f"--report_json {report_path} --output_dir {os.path.join(args.output_dir, 'artifacts')}"
    )
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
