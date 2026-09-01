"""NeuroLens explainability engine: trajectory retrieval + counterfactual steering.

Replaces static per-sample saliency (SHAP / Integrated Gradients) with two
dynamic mechanisms operating on the model's own 128-d normalized latent
trajectory z_t (see model.py, NeuroLensBackbone):

    - TrajectoryVectorDB: indexes historical rolling latent trajectories
      (k=12 consecutive 5-second windows -> 1 minute of state evolution)
      with FAISS, and scores a live trajectory's alignment against the
      nearest historical preictal and interictal runs (TAS).
    - CounterfactualSteeringEngine: given a high-risk live state, finds the
      minimal latent perturbation that would have made the model call it
      safe (projected gradient descent in latent space), and translates
      that perturbation back into a clinician-readable statement about
      channel-pair synchrony via the model's own Riemannian tangent space.

FAISS note: `faiss-gpu` and `faiss-cpu` both import as the top-level
module `faiss`; there is no separate importable name to fall back
between. The "GPU with fallback to CPU" behavior implemented here is
therefore a runtime capability check (CUDA resources available in this
faiss build) rather than an import-time choice.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when faiss is absent
    faiss = None
    _FAISS_AVAILABLE = False


# --------------------------------------------------------------------------
# Shared constants: the 18-channel bipolar montage and its clinical grouping.
#
# Duplicated (not imported) from dataset.py so this module can be deployed
# for real-time inference/explanation without pulling in dataset.py's mne
# dependency. Must stay in sync with dataset.BIPOLAR_MONTAGE.
# --------------------------------------------------------------------------

BIPOLAR_MONTAGE: List[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

# Standard clinical "double banana" bipolar-chain naming, used to translate
# a channel-pair delta into a scalp-region statement for the clinician.
CHANNEL_REGION: Dict[str, str] = {
    "FP1-F7": "Fronto-Temporal (Left)",
    "F7-T7": "Fronto-Temporal (Left)",
    "T7-P7": "Temporal (Left)",
    "P7-O1": "Temporo-Occipital (Left)",
    "FP1-F3": "Frontal (Left)",
    "F3-C3": "Fronto-Central (Left)",
    "C3-P3": "Centro-Parietal (Left)",
    "P3-O1": "Parieto-Occipital (Left)",
    "FP2-F4": "Frontal (Right)",
    "F4-C4": "Fronto-Central (Right)",
    "C4-P4": "Centro-Parietal (Right)",
    "P4-O2": "Parieto-Occipital (Right)",
    "FP2-F8": "Fronto-Temporal (Right)",
    "F8-T8": "Fronto-Temporal (Right)",
    "T8-P8": "Temporal (Right)",
    "P8-O2": "Temporo-Occipital (Right)",
    "FZ-CZ": "Fronto-Central (Midline)",
    "CZ-PZ": "Centro-Parietal (Midline)",
}

_CLASS_NAMES: Dict[int, str] = {0: "interictal", 1: "preictal"}


def _class_name(label: int) -> str:
    return _CLASS_NAMES.get(int(label), str(int(label)))


# --------------------------------------------------------------------------
# FAISS GPU-with-CPU-fallback helper
# --------------------------------------------------------------------------


def _to_gpu_if_available(cpu_index: "faiss.Index") -> Tuple["faiss.Index", bool]:
    """Try to move a freshly-built flat index onto GPU 0; fall back to CPU.

    Silently stays on CPU whenever: faiss isn't installed, this faiss build
    has no GPU support (faiss-cpu), or no CUDA device is visible.
    """
    if not _FAISS_AVAILABLE:
        return cpu_index, False
    try:
        n_gpus = faiss.get_num_gpus()
    except AttributeError:
        return cpu_index, False  # faiss-cpu build: GPU symbols don't exist
    if n_gpus <= 0:
        return cpu_index, False
    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        return gpu_index, True
    except Exception as exc:  # pragma: no cover - depends on local CUDA/faiss build
        warnings.warn(f"faiss GPU index construction failed ({exc}); falling back to CPU.")
        return cpu_index, False


# --------------------------------------------------------------------------
# Trajectory retrieval
# --------------------------------------------------------------------------


@dataclass
class TrajectoryMatch:
    """One historical trajectory retrieved for a live query."""

    global_id: int
    label: int
    similarity: float          # higher is always better, regardless of index metric
    metadata: Dict
    trajectory: np.ndarray     # [k, d]


@dataclass
class TrajectoryAlignmentResult:
    """Output of TrajectoryVectorDB.compute_trajectory_alignment_score."""

    tas_by_class: Dict[str, float]              # e.g. {"interictal": 0.41, "preictal": 0.87}
    per_match_tas: Dict[str, List[float]]        # per-match TAS within each class's top-K
    drift_velocity: float                        # mean ||z_l - z_{l-1}|| along the live trajectory
    drift_profile: np.ndarray                     # [k-1] per-step drift
    dominant_class: str                           # class the live trajectory aligns with most


class TrajectoryVectorDB:
    """FAISS-backed index of historical rolling latent trajectories.

    A trajectory is k consecutive per-window latent vectors (default
    k=12, i.e. 1 minute of 5-second windows). Trajectories are indexed
    separately per class (interictal / preictal) so that top-K retrieval
    against each class is guaranteed K results regardless of class
    imbalance (interictal windows vastly outnumber preictal ones under
    the labeling rules in dataset.py).

    Cosine similarity between two unit-norm trajectories, flattened to
    [k*d], is *exactly* the Trajectory Alignment Score of those two
    trajectories: since every z is unit-norm, each flattened vector has
    constant norm sqrt(k), so
        cos(flat_live, flat_hist) = (1/k) sum_l z_live^l . z_hist^l = TAS.
    An exact (flat) FAISS cosine index therefore retrieves precisely the
    top-K trajectories by TAS, with no approximation.
    """

    def __init__(self):
        self.k: Optional[int] = None
        self.d: Optional[int] = None
        self.metric: Optional[str] = None
        self._trajectories: Optional[np.ndarray] = None   # [N, k, d]
        self._labels: Optional[np.ndarray] = None          # [N]
        self._metadata: List[Dict] = []
        self._indices: Dict[int, "faiss.Index"] = {}
        self._ids_by_class: Dict[int, np.ndarray] = {}
        self._is_gpu: Dict[int, bool] = {}

    # -- construction -----------------------------------------------------

    def _flatten_and_normalize(self, trajectories: np.ndarray) -> np.ndarray:
        n, k, d = trajectories.shape
        flat = trajectories.reshape(n, k * d).astype(np.float32)
        if self.metric == "cosine":
            norms = np.clip(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12, None)
            flat = flat / norms
        return np.ascontiguousarray(flat)

    def _build_faiss_index(self, vectors: np.ndarray) -> "faiss.Index":
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for TrajectoryVectorDB (pip install faiss-gpu, or "
                "faiss-cpu as a fallback)."
            )
        dim = vectors.shape[1]
        if self.metric == "cosine":
            cpu_index = faiss.IndexFlatIP(dim)   # exact inner-product search
        elif self.metric == "l2":
            cpu_index = faiss.IndexFlatL2(dim)   # exact L2 search
        else:
            raise ValueError(f"metric must be 'cosine' or 'l2', got {self.metric!r}")
        cpu_index.add(vectors)
        index, is_gpu = _to_gpu_if_available(cpu_index)
        self._is_gpu[id(index)] = is_gpu
        return index

    def build_index(
        self,
        train_embeddings: Sequence[np.ndarray],
        labels: Sequence[np.ndarray],
        metadata: Sequence[Dict],
        k: int = 12,
        stride: int = 1,
        metric: str = "cosine",
    ) -> "TrajectoryVectorDB":
        """Build per-class FAISS indices of rolling k-window trajectories.

        Args:
            train_embeddings: one array [T_i, d] per temporally CONTIGUOUS
                chronological run of window embeddings (no excluded/skipped
                windows inside a run -- callers should split at any gap,
                e.g. the SPH/interictal-buffer exclusions in dataset.py,
                the same way loss.py's temporal-continuity term requires
                exact-stride adjacency).
            labels: matching sequence of int arrays [T_i], one label per
                window in each run.
            metadata: one dict per run (e.g. patient_id, file, seizure_idx);
                stored per trajectory with `sequence_idx`/`start_offset` added.
            k: trajectory length in windows (default 12 = 1 minute at 5s/window).
            stride: step between consecutive trajectory start offsets
                within a run; 1 gives maximally-overlapping rolling segments.
            metric: "cosine" (default; see class docstring for why this is
                exactly the TAS metric) or "l2".
        """
        if len(train_embeddings) != len(labels) or len(train_embeddings) != len(metadata):
            raise ValueError("train_embeddings, labels, and metadata must have equal length")

        all_traj, all_labels, all_meta = [], [], []
        for seq_idx, (emb_seq, lab_seq, meta) in enumerate(zip(train_embeddings, labels, metadata)):
            emb_seq = np.asarray(emb_seq, dtype=np.float32)
            lab_seq = np.asarray(lab_seq)
            if emb_seq.shape[0] != lab_seq.shape[0]:
                raise ValueError(f"sequence {seq_idx}: embeddings/labels length mismatch")
            T = emb_seq.shape[0]
            for start in range(0, T - k + 1, stride):
                segment = emb_seq[start : start + k]
                # The trajectory's label is that of its most recent (last)
                # window -- "what state is this 1-minute run of brain
                # activity currently in", which is what a live query needs
                # to be compared against apples-to-apples.
                seg_label = int(lab_seq[start + k - 1])
                seg_meta = dict(meta)
                seg_meta["sequence_idx"] = seq_idx
                seg_meta["start_offset"] = start
                all_traj.append(segment)
                all_labels.append(seg_label)
                all_meta.append(seg_meta)

        if not all_traj:
            raise ValueError(
                f"No trajectories of length k={k} could be built; every provided run is shorter than k."
            )

        trajectories = np.stack(all_traj, axis=0)  # [N, k, d]
        labels_arr = np.asarray(all_labels, dtype=np.int64)

        self.k = k
        self.d = int(trajectories.shape[-1])
        self.metric = metric
        self._trajectories = trajectories
        self._labels = labels_arr
        self._metadata = all_meta

        flat = self._flatten_and_normalize(trajectories)

        self._indices = {}
        self._ids_by_class = {}
        for cls in np.unique(labels_arr):
            cls_int = int(cls)
            cls_mask = labels_arr == cls
            self._ids_by_class[cls_int] = np.nonzero(cls_mask)[0].astype(np.int64)
            self._indices[cls_int] = self._build_faiss_index(flat[cls_mask])

        return self

    # -- persistence --------------------------------------------------------

    def save(self, dir_path: str) -> None:
        """Export this index to `dir_path`: raw trajectories/labels/metadata
        (numpy/JSON, the source of truth for reconstruction) plus one exact
        FAISS index file per class (GPU indices are copied back to CPU
        first, since FAISS's on-disk format is CPU-only)."""
        if self._trajectories is None:
            raise RuntimeError("build_index must be called before save")
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss is required to save a TrajectoryVectorDB index")

        os.makedirs(dir_path, exist_ok=True)
        np.save(os.path.join(dir_path, "trajectories.npy"), self._trajectories)
        np.save(os.path.join(dir_path, "labels.npy"), self._labels)
        with open(os.path.join(dir_path, "metadata.json"), "w") as f:
            json.dump(self._metadata, f, default=str)

        classes = [int(c) for c in self._indices.keys()]
        with open(os.path.join(dir_path, "manifest.json"), "w") as f:
            json.dump({"k": self.k, "d": self.d, "metric": self.metric, "classes": classes}, f)

        for cls, ids in self._ids_by_class.items():
            np.save(os.path.join(dir_path, f"ids_class_{cls}.npy"), ids)

        for cls, index in self._indices.items():
            cpu_index = faiss.index_gpu_to_cpu(index) if self._is_gpu.get(id(index), False) else index
            faiss.write_index(cpu_index, os.path.join(dir_path, f"index_class_{cls}.faiss"))

    @classmethod
    def load(cls, dir_path: str) -> "TrajectoryVectorDB":
        """Load an index previously written by :meth:`save`. GPU placement
        is retried per :func:`_to_gpu_if_available` (i.e. it need not match
        whatever device the index was saved from)."""
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss is required to load a TrajectoryVectorDB index")

        with open(os.path.join(dir_path, "manifest.json")) as f:
            manifest = json.load(f)

        db = cls()
        db.k = manifest["k"]
        db.d = manifest["d"]
        db.metric = manifest["metric"]
        db._trajectories = np.load(os.path.join(dir_path, "trajectories.npy"))
        db._labels = np.load(os.path.join(dir_path, "labels.npy"))
        with open(os.path.join(dir_path, "metadata.json")) as f:
            db._metadata = json.load(f)

        db._indices = {}
        db._ids_by_class = {}
        for cls_int in manifest["classes"]:
            db._ids_by_class[cls_int] = np.load(os.path.join(dir_path, f"ids_class_{cls_int}.npy"))
            cpu_index = faiss.read_index(os.path.join(dir_path, f"index_class_{cls_int}.faiss"))
            index, is_gpu = _to_gpu_if_available(cpu_index)
            db._indices[cls_int] = index
            db._is_gpu[id(index)] = is_gpu
        return db

    # -- retrieval ----------------------------------------------------------

    def query_nearest_trajectories(
        self, live_trajectory: Union[np.ndarray, torch.Tensor], top_k: int = 5
    ) -> Dict[str, List[TrajectoryMatch]]:
        """Retrieve the top-K nearest historical trajectory per class.

        Returns a dict keyed by class name (e.g. "interictal", "preictal"),
        each holding up to `top_k` :class:`TrajectoryMatch`, sorted best-first.
        """
        if not self._indices:
            raise RuntimeError("TrajectoryVectorDB.build_index must be called before querying")

        live = self._as_numpy_trajectory(live_trajectory)
        flat_query = self._flatten_and_normalize(live[None, :, :])

        results: Dict[str, List[TrajectoryMatch]] = {}
        for cls, index in self._indices.items():
            n_available = index.ntotal
            k_eff = min(top_k, n_available)
            class_name = _class_name(cls)
            if k_eff == 0:
                results[class_name] = []
                continue
            if k_eff < top_k:
                warnings.warn(
                    f"Only {k_eff} historical '{class_name}' trajectories available (top_k={top_k} requested)."
                )
            scores, local_ids = index.search(flat_query, k_eff)
            matches = []
            for score, local_id in zip(scores[0], local_ids[0]):
                if local_id < 0:
                    continue
                global_id = int(self._ids_by_class[cls][local_id])
                similarity = float(score) if self.metric == "cosine" else -float(score)
                matches.append(
                    TrajectoryMatch(
                        global_id=global_id,
                        label=cls,
                        similarity=similarity,
                        metadata=self._metadata[global_id],
                        trajectory=self._trajectories[global_id],
                    )
                )
            results[class_name] = matches
        return results

    def compute_trajectory_alignment_score(
        self,
        live_trajectory: Union[np.ndarray, torch.Tensor],
        historical_matches: Dict[str, List[TrajectoryMatch]],
    ) -> TrajectoryAlignmentResult:
        """Trajectory Alignment Score per retrieved class, plus drift velocity.

            TAS_class = (1/K) sum_k (1/L) sum_l cos(z_live^l, z_hist,k^l)

        computed directly from the retrieved trajectories (not merely
        recovered from FAISS distances), so it is exact even if a
        non-cosine index was used to retrieve `historical_matches`.
        Drift velocity is the mean per-step latent speed
        ||z_live^l - z_live^{l-1}||_2 along the live trajectory, a
        model-free proxy for how fast the current state is moving through
        latent space.
        """
        live = self._as_numpy_trajectory(live_trajectory)
        live_norm = live / np.clip(np.linalg.norm(live, axis=-1, keepdims=True), 1e-12, None)

        tas_by_class: Dict[str, float] = {}
        per_match_tas: Dict[str, List[float]] = {}
        for class_name, matches in historical_matches.items():
            if not matches:
                tas_by_class[class_name] = float("nan")
                per_match_tas[class_name] = []
                continue
            scores = []
            for m in matches:
                hist = np.asarray(m.trajectory, dtype=np.float32)
                hist_norm = hist / np.clip(np.linalg.norm(hist, axis=-1, keepdims=True), 1e-12, None)
                L = min(live_norm.shape[0], hist_norm.shape[0])
                cos_per_step = np.sum(live_norm[:L] * hist_norm[:L], axis=-1)
                scores.append(float(np.mean(cos_per_step)))
            per_match_tas[class_name] = scores
            tas_by_class[class_name] = float(np.mean(scores))

        diffs = np.diff(live_norm, axis=0)
        drift_profile = np.linalg.norm(diffs, axis=-1) if diffs.shape[0] > 0 else np.zeros(0, dtype=np.float32)
        drift_velocity = float(np.mean(drift_profile)) if drift_profile.size > 0 else 0.0

        valid = {c: v for c, v in tas_by_class.items() if not np.isnan(v)}
        dominant_class = max(valid, key=valid.get) if valid else "unknown"

        return TrajectoryAlignmentResult(
            tas_by_class=tas_by_class,
            per_match_tas=per_match_tas,
            drift_velocity=drift_velocity,
            drift_profile=drift_profile,
            dominant_class=dominant_class,
        )

    def _as_numpy_trajectory(self, live_trajectory: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(live_trajectory, torch.Tensor):
            live = live_trajectory.detach().cpu().numpy()
        else:
            live = np.asarray(live_trajectory)
        if live.ndim == 3:
            if live.shape[0] != 1:
                raise ValueError(f"expected a single trajectory, got batch of {live.shape[0]}")
            live = live[0]
        live = live.astype(np.float32)
        if self.k is not None and live.shape[0] != self.k:
            raise ValueError(f"live_trajectory has length {live.shape[0]}, expected k={self.k}")
        if self.d is not None and live.shape[-1] != self.d:
            raise ValueError(f"live_trajectory has dim {live.shape[-1]}, expected d={self.d}")
        return live


# --------------------------------------------------------------------------
# Counterfactual latent steering
# --------------------------------------------------------------------------


@dataclass
class CounterfactualResult:
    delta_z: torch.Tensor                    # [B, latent_dim] or [latent_dim]
    z_counterfactual: torch.Tensor
    initial_risk: torch.Tensor
    final_risk: torch.Tensor
    converged: torch.Tensor                  # bool, per batch item
    n_iters: int
    history: List[Dict[str, float]]
    delta_tangent: Optional[torch.Tensor] = None  # set only if raw_window was supplied


class CounterfactualSteeringEngine:
    """Finds the minimal latent perturbation that de-risks a live state.

        min_{delta_z} ||delta_z||_2^2   s.t.   f_risk(z_t + delta_z) < gamma

    solved with projected gradient descent: each step descends a risk
    penalty while infeasible, then descends ||delta_z||^2 (with a soft
    penalty pulling back toward feasibility) once a feasible point has been
    found; z_t + delta_z is re-projected onto the unit latent sphere after
    every step, since that sphere is the model's actual latent manifold.
    The smallest-norm feasible perturbation seen across all iterations is
    returned, not merely the last one.
    """

    def __init__(self, mc_samples_grad: int = 5, lagrange_lambda: float = 50.0):
        """
        Args:
            mc_samples_grad: number of differentiable stochastic forward
                passes through the model's (MC-Dropout) classifier head to
                average per PGD step, for a lower-variance risk gradient.
            lagrange_lambda: penalty weight applied to constraint violation
                once a feasible point has been reached, pulling norm-
                minimization steps back if they would leave the feasible set.
        """
        self.mc_samples_grad = mc_samples_grad
        self.lagrange_lambda = lagrange_lambda

    def _risk(self, model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
        logits = torch.stack(
            [model.classifier(z).squeeze(-1) for _ in range(self.mc_samples_grad)], dim=0
        )
        return torch.sigmoid(logits).mean(dim=0)

    def compute_counterfactual_perturbation(
        self,
        model: torch.nn.Module,
        current_latent: torch.Tensor,
        target_risk: float = 0.1,
        lr: float = 0.01,
        max_iter: int = 100,
        raw_window: Optional[torch.Tensor] = None,
    ) -> CounterfactualResult:
        """Latent-space PGD counterfactual search.

        Args:
            model: a NeuroLensBackbone (or any module exposing a
                differentiable `.classifier(z) -> logits` head consistent
                with `current_latent`'s space).
            current_latent: [latent_dim] or [B, latent_dim] unit-norm z_t.
            target_risk: gamma, the sigmoid-probability threshold defining
                the "safe" region f_risk(z) < gamma.
            lr: PGD step size.
            max_iter: number of PGD iterations.
            raw_window: optional raw EEG epoch ([C, T] or [B, C, T]) that
                produced `current_latent`. When given, this also computes
                an exact first-order pullback of delta_z onto the model's
                Riemannian tangent space (via vector-Jacobian product
                through `model.encoder`'s own submodules), returned as
                `delta_tangent` and consumable by
                `project_perturbation_to_channels`. Without it, channel-
                level attribution is not available (delta_z alone is not
                interpretable in channel space; see `project_perturbation_to_channels`).
        """
        was_training = model.training
        model.eval()  # MCDropout stays stochastic regardless; this just freezes any future train-only layers
        original_requires_grad = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)

        squeeze_output = current_latent.dim() == 1
        z0 = current_latent.detach().clone()
        if squeeze_output:
            z0 = z0.unsqueeze(0)
        B = z0.shape[0]
        device = z0.device

        try:
            initial_risk = self._risk(model, F.normalize(z0, p=2, dim=-1)).detach()

            delta = torch.zeros_like(z0, requires_grad=True)
            best_delta = torch.zeros_like(z0)
            best_norm = torch.full((B,), float("inf"), device=device)
            ever_feasible = torch.zeros(B, dtype=torch.bool, device=device)
            history: List[Dict[str, float]] = []

            def _update_best(delta_value: torch.Tensor, risk_value: torch.Tensor) -> torch.Tensor:
                # Records delta_value as the new best iff it is BOTH feasible
                # and smaller-norm than the best seen so far. Must be called
                # with the delta/risk pair exactly as evaluated together
                # (i.e. before any further in-place optimizer update to delta).
                feasible_now = risk_value < target_risk
                norm_now = delta_value.pow(2).sum(dim=-1).sqrt()
                better = feasible_now & (norm_now < best_norm)
                best_delta[better] = delta_value[better]
                best_norm[better] = norm_now[better]
                ever_feasible.logical_or_(feasible_now)
                return feasible_now

            optimizer = torch.optim.SGD([delta], lr=lr)
            it = 0
            for it in range(max_iter):
                optimizer.zero_grad()
                delta_before_step = delta.detach().clone()
                z_pert = F.normalize(z0 + delta, p=2, dim=-1)  # project back onto the unit-latent manifold
                risk = self._risk(model, z_pert)
                norm_sq = (delta ** 2).sum(dim=-1)
                feasible = risk < target_risk

                violation = torch.clamp(risk - target_risk, min=0.0)
                loss_per_sample = torch.where(
                    feasible,
                    norm_sq + self.lagrange_lambda * violation ** 2,
                    risk + 1e-3 * norm_sq,
                )
                loss_per_sample.sum().backward()
                optimizer.step()  # mutates delta in-place; delta_before_step/risk/feasible above are unaffected

                with torch.no_grad():
                    _update_best(delta_before_step, risk.detach())

                history.append(
                    {
                        "iter": it,
                        "mean_risk": risk.mean().item(),
                        "mean_norm": norm_sq.detach().sqrt().mean().item(),
                        "frac_feasible": feasible.float().mean().item(),
                    }
                )

            with torch.no_grad():
                # The loop above only ever scores delta *before* each step's
                # update (max_iter values: delta_0=0 .. delta_{max_iter-1});
                # score the final post-update delta too so it isn't missed.
                z_pert_final = F.normalize(z0 + delta, p=2, dim=-1)
                risk_final = self._risk(model, z_pert_final)
                _update_best(delta.detach(), risk_final)

            final_delta = torch.where(ever_feasible.unsqueeze(-1), best_delta, delta.detach())
            z_final = F.normalize(z0 + final_delta, p=2, dim=-1)
            final_risk = self._risk(model, z_final).detach()
        finally:
            for p, rg in zip(model.parameters(), original_requires_grad):
                p.requires_grad_(rg)
            model.train(was_training)

        delta_tangent = None
        if raw_window is not None:
            delta_tangent = self._pullback_delta_to_tangent(model, raw_window, final_delta)

        if squeeze_output:
            final_delta = final_delta.squeeze(0)
            z_final = z_final.squeeze(0)
            initial_risk = initial_risk.squeeze(0)
            final_risk = final_risk.squeeze(0)
            ever_feasible = ever_feasible.squeeze(0)

        return CounterfactualResult(
            delta_z=final_delta,
            z_counterfactual=z_final,
            initial_risk=initial_risk,
            final_risk=final_risk,
            converged=ever_feasible,
            n_iters=it + 1,
            history=history,
            delta_tangent=delta_tangent,
        )

    def _pullback_delta_to_tangent(
        self, model: torch.nn.Module, raw_window: torch.Tensor, delta_z: torch.Tensor
    ) -> torch.Tensor:
        """Vector-Jacobian-product pullback of delta_z onto tangent space.

        Re-runs the encoder's own forward composition (its public
        submodules: tangent_projector -> scale branches -> fuse ->
        latent_head), but with the spatial-scale tangent vectors detached
        into a fresh leaf tensor, so `torch.autograd.grad` can compute
        d(z)/d(tangent)^T . delta_z exactly -- the local first-order
        direction, in tangent space, that a change of delta_z in z-space
        corresponds to. This uses the model's own trained weights (no
        external/learned surrogate), evaluated at the original latent
        state (a standard choice for gradient-based attribution).
        """
        encoder = model.encoder
        x = raw_window
        if x.dim() == 2:
            x = x.unsqueeze(0)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                n_patches = x.shape[-1] // encoder.spatial_patch_len
                spatial_patches = (
                    x[:, :, : n_patches * encoder.spatial_patch_len]
                    .reshape(x.shape[0], x.shape[1], n_patches, encoder.spatial_patch_len)
                    .permute(0, 2, 1, 3)
                )
                tangent0, _ = encoder.tangent_projector(spatial_patches)

            with torch.enable_grad():
                tangent = tangent0.detach().clone().requires_grad_(True)
                scale_summaries = []
                for i, branch in enumerate(encoder.branches):
                    tangent_in = tangent if i == encoder.spatial_scale_idx else None
                    scale_summaries.append(branch(x, tangent_in))
                fused = encoder.fuse(torch.cat(scale_summaries, dim=-1))
                z = F.normalize(model.latent_head(fused), p=2, dim=-1)

                grad_outputs = delta_z if delta_z.dim() == z.dim() else delta_z.unsqueeze(0)
                (delta_tangent,) = torch.autograd.grad(
                    outputs=z, inputs=tangent, grad_outputs=grad_outputs, retain_graph=False
                )
        finally:
            model.train(was_training)

        return delta_tangent.mean(dim=1).detach()  # pool over sub-patches -> [B, tangent_dim]

    def project_perturbation_to_channels(
        self, tangent_projector: torch.nn.Module, delta_z: Union[torch.Tensor, np.ndarray], top_k: int = 3
    ) -> Dict:
        """Translate a tangent-space delta into a clinician-readable statement.

        `delta_z` must be a tangent-space delta -- either flat
        `[tangent_projector.out_dim]` (as produced by un-vectorizing) or a
        symmetric `[C, C]` matrix (as produced directly by
        `_pullback_delta_to_tangent` before un-vectorization is undone
        here); a raw 128-d latent-space delta is not directly
        interpretable in channel space (the encoder that connects them is
        nonlinear) -- obtain a tangent-space delta first by calling
        `compute_counterfactual_perturbation(..., raw_window=...)` and
        passing its `delta_tangent` here.

        Returns a dict with per-channel-pair deltas, a region-level
        aggregate (using the standard clinical bipolar-chain names, e.g.
        "Fronto-Temporal"), and a one-sentence narrative for the dominant
        region.
        """
        C = tangent_projector.n_channels
        out_dim = tangent_projector.out_dim
        delta = delta_z.detach().cpu() if torch.is_tensor(delta_z) else torch.as_tensor(np.asarray(delta_z))
        delta = delta.float()

        # Normalize to a single, unbatched representation: either a flat
        # tangent vector [out_dim] or a symmetric matrix [C, C].
        if delta.dim() == 1 and delta.shape[0] == out_dim:
            vec, mat_in = delta, None
        elif delta.dim() == 2 and delta.shape[-1] == out_dim and delta.shape != (C, C):
            vec, mat_in = delta[0], None  # [B, out_dim] -> first batch item
        elif delta.dim() == 2 and delta.shape == (C, C):
            vec, mat_in = None, delta
        elif delta.dim() == 3 and delta.shape[-2:] == (C, C):
            vec, mat_in = None, delta[0]  # [B, C, C] -> first batch item
        else:
            raise ValueError(
                f"delta_z must be a flat tangent vector of length {out_dim} (optionally batched) "
                f"or a symmetric ({C},{C}) matrix (optionally batched); got shape {tuple(delta.shape)}. "
                "Pass CounterfactualResult.delta_tangent (from compute_counterfactual_perturbation "
                "called with raw_window=...), not the raw 128-d latent delta_z."
            )

        if vec is not None:
            triu_row = tangent_projector.triu_row
            triu_col = tangent_projector.triu_col
            vec_scale = tangent_projector.vec_scale.to(vec.dtype)
            raw_vals = vec / vec_scale
            mat = torch.zeros(C, C, dtype=vec.dtype)
            mat[triu_row, triu_col] = raw_vals
            mat = mat + mat.T - torch.diag(torch.diag(mat))
        else:
            mat = 0.5 * (mat_in + mat_in.T)  # symmetrize defensively

        mat_np = mat.numpy()
        channel_power_delta = {BIPOLAR_MONTAGE[i]: float(mat_np[i, i]) for i in range(C)}

        pairs: List[Tuple[str, str, float]] = []
        for i in range(C):
            for j in range(i + 1, C):
                pairs.append((BIPOLAR_MONTAGE[i], BIPOLAR_MONTAGE[j], float(mat_np[i, j])))
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)

        region_delta: Dict[str, float] = {}
        for i in range(C):
            region = CHANNEL_REGION[BIPOLAR_MONTAGE[i]]
            region_delta[region] = region_delta.get(region, 0.0) + mat_np[i, i]
        for ci, cj, v in pairs:
            r_i, r_j = CHANNEL_REGION[ci], CHANNEL_REGION[cj]
            region_delta[r_i] = region_delta.get(r_i, 0.0) + 0.5 * v
            region_delta[r_j] = region_delta.get(r_j, 0.0) + 0.5 * v
        ranked_regions = sorted(region_delta.items(), key=lambda kv: abs(kv[1]), reverse=True)

        top_region, top_value = ranked_regions[0]
        direction = "decrease" if top_value < 0 else "increase"
        narrative = (
            f"{'Reducing' if direction == 'decrease' else 'Increasing'} synchrony/power in the "
            f"{top_region} channels by {abs(top_value):.3f} (log-covariance units) is the dominant "
            "step toward the stable interictal attractor identified by this counterfactual."
        )

        return {
            "top_channel_pairs": pairs[:top_k],
            "channel_power_delta": channel_power_delta,
            "region_delta": dict(ranked_regions),
            "dominant_region": top_region,
            "dominant_direction": direction,
            "narrative": narrative,
        }
