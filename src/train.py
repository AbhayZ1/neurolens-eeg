"""NeuroLens training pipeline: per-patient LOSO seizure-prediction models.

For every (patient, held-out seizure) LOSO fold this:

    Phase A -- contrastive pretraining of the full backbone with
               DynamicalSupConLoss (SupCon + temporal-continuity only;
               no classification signal yet).
    Phase B -- linear probing (default: encoder + latent_head frozen,
               only the classifier head trained) or full fine-tuning,
               with DynamicalSupConLoss's BCE term added; the best epoch
               by validation AUPRC is kept.
    Calibration -- a scalar temperature is fit (Guo et al., 2017) to
               minimize NLL of the MC-Dropout-averaged validation
               probabilities, for later ECE reduction at inference time.
    Export -- the fold's trained encoder is used to embed every
               temporally-contiguous training run, and a per-class FAISS
               TrajectoryVectorDB is built and saved alongside the
               checkpoint.

Usage:
    python train.py --data_dir /path/to/chb-mit-scalp-eeg-database-1.0.0 \
        --patients chb01 chb02 chb03 --output_dir checkpoints
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow `python path/to/train.py` from anywhere

from dataset import (
    CHBMITChronologicalDataset,
    FileRecord,
    WindowRecord,
    WINDOW_SEC,
    build_patient_windows,
    get_loso_splits,
    group_contiguous_runs,
)
from loss import DynamicalSupConLoss
from model import NeuroLensBackbone
from xai_engine import TrajectoryVectorDB


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "checkpoints"
    patients: Sequence[str] = field(default_factory=lambda: ["chb01", "chb02", "chb03"])
    device: str = "auto"
    seed: int = 42

    # model architecture (must match model.NeuroLensBackbone's constructor)
    n_channels: int = 18
    seq_len: int = 1280
    patch_lens: Sequence[int] = field(default_factory=lambda: [128, 256])
    spatial_scale_idx: int = 0
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout_p: float = 0.3
    latent_dim: int = 128

    # data loading
    batch_size: int = 32
    val_fraction: float = 0.15
    num_workers: int = 0
    raw_cache_size: int = 2

    # Phase A: contrastive pretraining
    epochs_pretrain: int = 10
    lr_pretrain: float = 1e-3
    supcon_temperature: float = 0.1
    lambda_temporal: float = 0.5

    # Phase B: linear probe / fine-tune
    epochs_finetune: int = 15
    lr_finetune: float = 1e-4
    freeze_encoder_phase_b: bool = True
    grad_clip_norm: float = 5.0

    # post-hoc temperature scaling
    temp_scale_mc_samples: int = 20
    temp_scale_lr: float = 0.01
    temp_scale_max_iter: int = 200

    # FAISS trajectory index export
    trajectory_k: int = 12
    trajectory_stride: int = 1


def _model_kwargs(config: TrainConfig) -> Dict:
    """NeuroLensBackbone constructor kwargs, single source of truth so the
    exact architecture used at train time can be reconstructed at eval time."""
    return dict(
        n_channels=config.n_channels,
        seq_len=config.seq_len,
        patch_lens=tuple(config.patch_lens),
        spatial_scale_idx=config.spatial_scale_idx,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout_p=config.dropout_p,
        latent_dim=config.latent_dim,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# A dataset wrapper that also yields (timestamp, sequence_id), needed by
# DynamicalSupConLoss's temporal-continuity adjacency check but not part
# of CHBMITChronologicalDataset's own (x, label) contract.
# --------------------------------------------------------------------------


class _TrajectoryAwareDataset(Dataset):
    def __init__(self, base: CHBMITChronologicalDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        w = self.base.windows[idx]
        return x, y, w.abs_start_sec, w.file_idx


def _make_metadata_loader(loader: DataLoader, shuffle: bool) -> DataLoader:
    wrapped = _TrajectoryAwareDataset(loader.dataset)
    return DataLoader(wrapped, batch_size=loader.batch_size, shuffle=shuffle, num_workers=loader.num_workers)


# --------------------------------------------------------------------------
# Temperature scaling (Guo et al., 2017)
# --------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """Single learned scalar T > 0, applied as logits / T."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 200
) -> TemperatureScaler:
    """Fit T minimizing BCE (equivalently NLL) of sigmoid(logits / T) on `labels`."""
    scaler = TemperatureScaler().to(logits.device)
    labels = labels.float()
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = bce(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


@torch.no_grad()
def _collect_mc_logits(
    model: NeuroLensBackbone, loader: DataLoader, device: torch.device, num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MC-Dropout-averaged probability, converted back to logit space, per
    validation window -- what temperature scaling should actually calibrate,
    since it's what's reported to the clinician at inference time."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        mc = model.get_mc_prediction(x, num_samples=num_samples)
        mean_prob = mc["mean_prob"].clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(mean_prob / (1.0 - mean_prob))
        all_logits.append(logit.cpu())
        all_labels.append(y)
    if not all_logits:
        return torch.empty(0), torch.empty(0)
    return torch.cat(all_logits), torch.cat(all_labels)


# --------------------------------------------------------------------------
# Epoch loops
# --------------------------------------------------------------------------


def _pretrain_epoch(
    model: NeuroLensBackbone,
    loader: DataLoader,
    loss_fn: DynamicalSupConLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y, ts, sid in loader:
        x, y, ts, sid = x.to(device), y.to(device), ts.to(device), sid.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss_dict = loss_fn(out["latent_vector"], y, timestamps=ts, sequence_ids=sid)
        loss_dict["loss"].backward()
        if grad_clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total += loss_dict["loss"].item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def _finetune_epoch(
    model: NeuroLensBackbone,
    loader: DataLoader,
    loss_fn: DynamicalSupConLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> Dict[str, float]:
    model.train()
    total, total_bce, n = 0.0, 0.0, 0
    for x, y, ts, sid in loader:
        x, y, ts, sid = x.to(device), y.to(device), ts.to(device), sid.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss_dict = loss_fn(
            out["latent_vector"], y, seizure_logits=out["seizure_logits"], timestamps=ts, sequence_ids=sid
        )
        loss_dict["loss"].backward()
        if grad_clip_norm:
            trainable = [p for p in model.parameters() if p.requires_grad]
            nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
        optimizer.step()
        total += loss_dict["loss"].item() * x.size(0)
        total_bce += loss_dict["bce_loss"].item() * x.size(0)
        n += x.size(0)
    return {"loss": total / max(n, 1), "bce_loss": total_bce / max(n, 1)}


@torch.no_grad()
def _quick_eval_auprc(model: NeuroLensBackbone, loader: DataLoader, device: torch.device) -> float:
    """Fast (single stochastic forward pass, not MC-averaged) AUPRC for
    per-epoch model selection; full MC-averaged metrics belong in evaluate.py."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)["seizure_logits"]
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    if not probs:
        return float("nan")
    probs_arr = np.concatenate(probs)
    labels_arr = np.concatenate(labels)
    if len(np.unique(labels_arr)) < 2:
        return float("nan")
    return float(average_precision_score(labels_arr, probs_arr))


# --------------------------------------------------------------------------
# Per-fold training
# --------------------------------------------------------------------------


@dataclass
class FoldResult:
    patient_id: str
    test_seizure_idx: int
    model: NeuroLensBackbone
    temperature: float
    best_val_auprc: float
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    history: List[Dict]


def train_one_fold(
    patient_id: str,
    test_seizure_idx: int,
    config: TrainConfig,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> FoldResult:
    log = logger or logging.getLogger("neurolens.train")

    train_loader, val_loader, test_loader = get_loso_splits(
        patient_id,
        config.data_dir,
        test_seizure_idx,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        num_workers=config.num_workers,
        raw_cache_size=config.raw_cache_size,
    )
    train_loader_meta = _make_metadata_loader(train_loader, shuffle=True)

    model = NeuroLensBackbone(**_model_kwargs(config)).to(device)
    history: List[Dict] = []

    # ---- Phase A: contrastive pretraining -------------------------------
    pretrain_loss_fn = DynamicalSupConLoss(
        temperature=config.supcon_temperature,
        lambda_temporal=config.lambda_temporal,
        window_stride_sec=float(WINDOW_SEC),
    )
    optimizer_a = torch.optim.AdamW(model.parameters(), lr=config.lr_pretrain)
    for epoch in range(config.epochs_pretrain):
        loss = _pretrain_epoch(model, train_loader_meta, pretrain_loss_fn, optimizer_a, device, config.grad_clip_norm)
        log.info(f"[{patient_id} fold{test_seizure_idx}] pretrain {epoch + 1}/{config.epochs_pretrain} loss={loss:.4f}")
        history.append({"phase": "pretrain", "epoch": epoch, "loss": loss})

    # ---- Phase B: linear probe / fine-tune -------------------------------
    if config.freeze_encoder_phase_b:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        for p in model.latent_head.parameters():
            p.requires_grad_(False)

    finetune_loss_fn = DynamicalSupConLoss(
        temperature=config.supcon_temperature,
        lambda_temporal=config.lambda_temporal,
        window_stride_sec=float(WINDOW_SEC),
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_b = torch.optim.AdamW(trainable_params, lr=config.lr_finetune)

    best_auprc = float("-inf")
    best_state = None
    for epoch in range(config.epochs_finetune):
        stats = _finetune_epoch(model, train_loader_meta, finetune_loss_fn, optimizer_b, device, config.grad_clip_norm)
        val_auprc = _quick_eval_auprc(model, val_loader, device)
        log.info(
            f"[{patient_id} fold{test_seizure_idx}] finetune {epoch + 1}/{config.epochs_finetune} "
            f"loss={stats['loss']:.4f} bce={stats['bce_loss']:.4f} val_auprc={val_auprc:.4f}"
        )
        history.append({"phase": "finetune", "epoch": epoch, **stats, "val_auprc": val_auprc})
        if not math.isnan(val_auprc) and val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        log.warning(f"[{patient_id} fold{test_seizure_idx}] val_auprc was never defined; keeping final-epoch weights")
        best_auprc = float("nan")

    # restore full trainability regardless of freeze_encoder_phase_b, so the
    # returned model is in a normal state for downstream use (embedding
    # extraction, further fine-tuning, etc.)
    for p in model.parameters():
        p.requires_grad_(True)

    # ---- Post-hoc temperature scaling on the validation set -------------
    model.eval()
    val_logits, val_labels = _collect_mc_logits(model, val_loader, device, config.temp_scale_mc_samples)
    if val_logits.numel() > 0 and len(torch.unique(val_labels)) > 1:
        scaler = fit_temperature(val_logits, val_labels, lr=config.temp_scale_lr, max_iter=config.temp_scale_max_iter)
        temperature = float(scaler.temperature.item())
    else:
        log.warning(f"[{patient_id} fold{test_seizure_idx}] validation set has a single class; skipping temperature scaling (T=1.0)")
        temperature = 1.0
    log.info(f"[{patient_id} fold{test_seizure_idx}] fitted calibration temperature T={temperature:.4f}")

    return FoldResult(
        patient_id=patient_id,
        test_seizure_idx=test_seizure_idx,
        model=model,
        temperature=temperature,
        best_val_auprc=best_auprc,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        history=history,
    )


# --------------------------------------------------------------------------
# FAISS trajectory index export
# --------------------------------------------------------------------------


@torch.no_grad()
def _embed_run(
    model: NeuroLensBackbone,
    patient_id: str,
    data_dir: str,
    file_records: List[FileRecord],
    run_windows: List[WindowRecord],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed one contiguous run of windows, in order, via the model's own
    (x, label) Dataset/DataLoader machinery -- avoids duplicating EDF
    loading/preprocessing logic here."""
    ds = CHBMITChronologicalDataset(patient_id, data_dir, run_windows, file_records)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    embeddings, labels = [], []
    for x, y in loader:
        x = x.to(device)
        z = model(x)["latent_vector"].cpu().numpy()
        embeddings.append(z)
        labels.append(y.numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def build_faiss_index_for_fold(
    model: NeuroLensBackbone,
    patient_id: str,
    data_dir: str,
    train_loader: DataLoader,
    device: torch.device,
    k: int = 12,
    stride: int = 1,
) -> TrajectoryVectorDB:
    """Embed every contiguous training run and index the resulting rolling
    k-window trajectories, per-class, in a TrajectoryVectorDB."""
    ds: CHBMITChronologicalDataset = train_loader.dataset
    runs = group_contiguous_runs(ds.windows)

    embeddings_seqs, label_seqs, meta_seqs = [], [], []
    for run in runs:
        if len(run) < k:
            continue
        emb, lab = _embed_run(model, patient_id, data_dir, ds.file_records, run, device)
        embeddings_seqs.append(emb)
        label_seqs.append(lab)
        meta_seqs.append(
            {
                "patient_id": patient_id,
                "file_idx": run[0].file_idx,
                "seizure_idx": run[0].seizure_idx,
                "n_windows": len(run),
            }
        )

    db = TrajectoryVectorDB()
    db.build_index(embeddings_seqs, label_seqs, meta_seqs, k=k, stride=stride, metric="cosine")
    return db


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, required=True, help="Root of the CHB-MIT database (contains chb01/, chb02/, ...).")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--patients", type=str, nargs="+", default=["chb01", "chb02", "chb03"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--epochs_pretrain", type=int, default=10)
    p.add_argument("--lr_pretrain", type=float, default=1e-3)
    p.add_argument("--supcon_temperature", type=float, default=0.1)
    p.add_argument("--lambda_temporal", type=float, default=0.5)

    p.add_argument("--epochs_finetune", type=int, default=15)
    p.add_argument("--lr_finetune", type=float, default=1e-4)
    p.add_argument(
        "--no_freeze_encoder",
        action="store_true",
        help="Fine-tune the whole network in Phase B instead of linear-probing the frozen encoder.",
    )

    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--dropout_p", type=float, default=0.3)
    p.add_argument("--latent_dim", type=int, default=128)

    p.add_argument("--trajectory_k", type=int, default=12)
    p.add_argument("--trajectory_stride", type=int, default=1)
    p.add_argument(
        "--max_folds_per_patient",
        type=int,
        default=None,
        help="Cap the number of LOSO folds trained per patient (useful for smoke tests).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("neurolens.train")

    set_seed(args.seed)
    device = resolve_device(args.device)
    log.info(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        patients=args.patients,
        device=str(device),
        seed=args.seed,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        epochs_pretrain=args.epochs_pretrain,
        lr_pretrain=args.lr_pretrain,
        supcon_temperature=args.supcon_temperature,
        lambda_temporal=args.lambda_temporal,
        epochs_finetune=args.epochs_finetune,
        lr_finetune=args.lr_finetune,
        freeze_encoder_phase_b=not args.no_freeze_encoder,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout_p=args.dropout_p,
        latent_dim=args.latent_dim,
        trajectory_k=args.trajectory_k,
        trajectory_stride=args.trajectory_stride,
    )

    summary: List[Dict] = []
    for patient_id in config.patients:
        try:
            _, _, all_seizures = build_patient_windows(patient_id, config.data_dir)
        except Exception as exc:
            log.error(f"Skipping patient {patient_id}: failed to build timeline ({exc})")
            continue

        n_seizures = len(all_seizures)
        n_folds = n_seizures if args.max_folds_per_patient is None else min(n_seizures, args.max_folds_per_patient)
        log.info(f"Patient {patient_id}: {n_seizures} seizures -> training {n_folds} LOSO fold(s)")

        for test_seizure_idx in range(n_folds):
            fold_id = f"{patient_id}_seizure{test_seizure_idx}"
            log.info(f"=== Training fold {fold_id} ===")
            try:
                result = train_one_fold(patient_id, test_seizure_idx, config, device, logger=log)
            except RuntimeError as exc:
                log.warning(f"Skipping fold {fold_id}: {exc}")
                continue

            ckpt_path = os.path.join(args.output_dir, f"{fold_id}.pt")
            torch.save(
                {
                    "model_state_dict": result.model.state_dict(),
                    "model_kwargs": _model_kwargs(config),
                    "temperature": result.temperature,
                    "val_auprc": result.best_val_auprc,
                    "patient_id": patient_id,
                    "test_seizure_idx": test_seizure_idx,
                    "history": result.history,
                },
                ckpt_path,
            )
            log.info(f"Saved checkpoint: {ckpt_path} (val_auprc={result.best_val_auprc:.4f})")

            index_dir = os.path.join(args.output_dir, f"{fold_id}_faiss_index")
            index_export_ok = True
            try:
                db = build_faiss_index_for_fold(
                    result.model,
                    patient_id,
                    config.data_dir,
                    result.train_loader,
                    device,
                    k=config.trajectory_k,
                    stride=config.trajectory_stride,
                )
                db.save(index_dir)
                log.info(f"Exported FAISS trajectory index: {index_dir}")
            except Exception as exc:
                log.warning(f"Failed to export FAISS index for {fold_id}: {exc}")
                index_export_ok = False

            summary.append(
                {
                    "fold_id": fold_id,
                    "patient_id": patient_id,
                    "test_seizure_idx": test_seizure_idx,
                    "val_auprc": result.best_val_auprc,
                    "temperature": result.temperature,
                    "checkpoint": ckpt_path,
                    "faiss_index_dir": index_dir if index_export_ok else None,
                    "n_train_windows": len(result.train_loader.dataset),
                    "n_val_windows": len(result.val_loader.dataset),
                    "n_test_windows": len(result.test_loader.dataset),
                }
            )

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"config": asdict(config), "folds": summary}, f, indent=2)
    log.info(f"Training complete. {len(summary)} fold(s) trained. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
