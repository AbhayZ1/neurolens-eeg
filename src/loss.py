"""Loss functions for NeuroLens trajectory-aware representation learning."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def _zero_but_connected(z: torch.Tensor) -> torch.Tensor:
    """A scalar 0.0 that stays attached to z's autograd graph.

    Used for "no valid pairs this batch" fallbacks: unlike z.new_zeros(()),
    which is a fresh constant with no grad_fn, `0.0 * z.sum()` is exactly
    zero-valued and contributes exactly zero gradient everywhere, but keeps
    a real (if trivial) graph connection back to z. That matters when this
    term is the *only* contributor to a training step's total loss (e.g.
    Phase A contrastive pretraining, which has no BCE fallback term): a
    fully-disconnected zero there makes `.backward()` raise
    "does not require grad and does not have a grad_fn" instead of just
    contributing nothing for that step, which is the intended degenerate
    behavior.
    """
    return 0.0 * z.sum()


class DynamicalSupConLoss(nn.Module):
    """Supervised Contrastive Loss + temporal trajectory smoothness + BCE.

        L_SupCon   = mean over anchors i with |P(i)| > 0 of
                     -1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p/tau) / sum_{a in A(i)} exp(z_i.z_a/tau) )

        L_temporal = mean over consecutive, same-recording, INTERICTAL
                     pairs (t-1, t) in the batch of || z_t - z_{t-1} ||_2^2

        L_total    = L_SupCon + lambda_temporal * L_temporal + L_BCE

    z is assumed L2-normalized (as produced by NeuroLensBackbone), so
    z_i . z_p is cosine similarity in [-1, 1]. A(i) is every other sample
    in the batch (standard SupCon "all" denominator).

    The temporal term is deliberately restricted to pairs the caller marks
    as temporally adjacent AND interictal on both ends -- it penalizes
    jagged jumps during steady interictal dynamics without discouraging
    the latent trajectory from moving during an actual regime change
    (preictal onset), which is the behavior the representation is
    supposed to capture. Adjacency is provided via ``timestamps``
    (absolute window start time, seconds) and optional ``sequence_ids``
    (e.g. file/recording index) rather than batch position, since training
    batches are typically shuffled and are not chronologically ordered.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        lambda_temporal: float = 0.5,
        window_stride_sec: float = 5.0,
        interictal_label: int = 0,
        time_tol_sec: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.lambda_temporal = lambda_temporal
        self.window_stride_sec = window_stride_sec
        self.interictal_label = interictal_label
        self.time_tol_sec = time_tol_sec
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def supervised_contrastive(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """z: [B, D] (L2-normalized), labels: [B] int class labels."""
        B = z.shape[0]
        if B < 2:
            return _zero_but_connected(z)

        labels = labels.view(-1, 1)
        same_class = torch.eq(labels, labels.T)                 # [B, B]
        self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        positive_mask = same_class & (~self_mask)

        sim = torch.matmul(z, z.T) / self.temperature            # [B, B]
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # log-sum-exp stability (cancels exactly)
        exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
        denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(self.eps)  # sum over A(i)
        log_prob = sim - torch.log(denom)                        # log( exp(sim_ip) / sum_a exp(sim_ia) )

        pos_count = positive_mask.sum(dim=1)                     # |P(i)|
        valid = pos_count > 0
        if not torch.any(valid):
            return _zero_but_connected(z)

        pos_log_prob_sum = (log_prob * positive_mask.float()).sum(dim=1)
        per_anchor_loss = -pos_log_prob_sum[valid] / pos_count[valid].float()
        return per_anchor_loss.mean()

    def temporal_continuity(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        timestamps: Optional[torch.Tensor],
        sequence_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Penalize ||z_t - z_{t-1}||^2 over consecutive interictal pairs found in the batch."""
        B = z.shape[0]
        if timestamps is None or B < 2:
            return _zero_but_connected(z)

        t = timestamps.view(-1, 1).float()
        delta = t.T - t  # delta[i, j] = timestamps[j] - timestamps[i]
        consecutive = torch.isclose(
            delta, torch.full_like(delta, self.window_stride_sec), atol=self.time_tol_sec
        )

        is_interictal = (labels == self.interictal_label).view(-1, 1)
        mask = consecutive & is_interictal & is_interictal.T

        if sequence_ids is not None:
            sid = sequence_ids.view(-1, 1)
            mask = mask & torch.eq(sid, sid.T)

        mask = mask & (~torch.eye(B, dtype=torch.bool, device=z.device))
        n_pairs = mask.sum()
        if n_pairs == 0:
            return _zero_but_connected(z)

        sq_dist = torch.cdist(z, z, p=2.0) ** 2  # [B, B], sq_dist[i, j] = ||z_i - z_j||^2
        return (sq_dist * mask.float()).sum() / n_pairs.float()

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        seizure_logits: Optional[torch.Tensor] = None,
        bce_labels: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        sequence_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:
            z: [B, D] L2-normalized latent states.
            labels: [B] int class labels (also used to gate the interictal
                temporal-continuity mask).
            seizure_logits: [B] raw classifier logits; if None, the BCE
                term is omitted (e.g. during contrastive-only pretraining).
            bce_labels: [B] binary targets for BCE; defaults to `labels`.
            timestamps: [B] absolute window start times (seconds); if
                None, the temporal-continuity term is omitted.
            sequence_ids: [B] recording/file identifiers, required to
                avoid treating windows from different recordings as
                temporally adjacent just because their timestamps happen
                to differ by exactly one window stride.
        """
        supcon_loss = self.supervised_contrastive(z, labels)
        temporal_loss = self.temporal_continuity(z, labels, timestamps, sequence_ids)
        total = supcon_loss + self.lambda_temporal * temporal_loss

        bce_loss = None
        if seizure_logits is not None:
            target = (bce_labels if bce_labels is not None else labels).float()
            bce_loss = self.bce(seizure_logits, target)
            total = total + bce_loss

        return {
            "loss": total,
            "supcon_loss": supcon_loss.detach(),
            "temporal_loss": temporal_loss.detach(),
            "bce_loss": bce_loss.detach() if bce_loss is not None else None,
        }
