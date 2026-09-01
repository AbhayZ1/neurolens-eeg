"""NeuroLens representation-learning backbone.

Spatial-temporal architecture over 18-channel, 5-second EEG epochs
(matches ``dataset.py``: ``N_CHANNELS=18``, ``N_SAMPLES=1280`` at 256 Hz):

    1. RiemannianTangentProjector -- each epoch is split into short
       sub-patches; a spatial sample-covariance matrix is estimated per
       patch and mapped onto the Log-Euclidean tangent space at the
       patch sequence's own Frechet mean, then vectorized.
    2. PatchTSTEncoder -- a multi-scale patch transformer that fuses
       those tangent-space (spatial) tokens with raw-signal (temporal)
       tokens and summarizes each scale via attention pooling.
    3. NeuroLensBackbone -- projects the fused representation to a unit-
       norm 128-d latent state z_t and a binary seizure-risk logit, with
       Monte Carlo Dropout active in both training and inference for
       epistemic uncertainty estimation.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defaults correspond to dataset.py: FS=256, WINDOW_SEC=5, N_CHANNELS=18.
DEFAULT_N_CHANNELS = 18
DEFAULT_SEQ_LEN = 1280


# --------------------------------------------------------------------------
# Monte Carlo Dropout
# --------------------------------------------------------------------------


class MCDropout(nn.Module):
    """Dropout that stays stochastic in both train() and eval() mode.

    Standard nn.Dropout becomes a no-op under model.eval(), which is
    exactly wrong for Monte Carlo Dropout: epistemic uncertainty requires
    the same stochastic forward pass at inference time, repeated and
    averaged. This module always calls F.dropout with training=True,
    independent of self.training.
    """

    def __init__(self, p: float = 0.3):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)

    def extra_repr(self) -> str:
        return f"p={self.p} (always active)"


# --------------------------------------------------------------------------
# Module 1: Riemannian tangent-space projection
# --------------------------------------------------------------------------


class RiemannianTangentProjector(nn.Module):
    """Maps per-patch spatial covariance matrices onto a Log-Euclidean
    tangent space and vectorizes them.

    For each patch p (samples x_p in R^{C x L}):

        C_p = (1 / (L - 1)) x_p x_p^T                      (sample covariance)

    C_p is regularized (diagonal shrinkage + eigenvalue condition-number
    clamping) before taking a matrix logarithm via eigendecomposition:

        C_p = U diag(lambda) U^T
        Logm(C_p) = U diag(log lambda_clamped) U^T

    Under the Log-Euclidean metric the tangent space is globally flat and
    coincides with the space of symmetric matrices via Logm, so the
    Frechet (geometric) mean of a set {C_p} is simply

        Frechet_mean_logm = mean_p Logm(C_p),   Frechet_mean = expm(Frechet_mean_logm)

    and the tangent vector of C_p *at* that Frechet mean is exactly the
    mean-centered log-map ``Logm(C_p) - Frechet_mean_logm``. Centered
    tangent matrices are then vectorized via the standard half-vectorization
    that preserves the Frobenius inner product (off-diagonal entries scaled
    by sqrt(2)).
    """

    def __init__(self, n_channels: int = DEFAULT_N_CHANNELS, eps: float = 1e-4, max_cond: float = 1e4):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.max_cond = max_cond

        triu_row, triu_col = torch.triu_indices(n_channels, n_channels)
        self.register_buffer("triu_row", triu_row, persistent=False)
        self.register_buffer("triu_col", triu_col, persistent=False)
        off_diag = (triu_row != triu_col)
        vec_scale = torch.where(off_diag, torch.tensor(2.0 ** 0.5), torch.tensor(1.0))
        self.register_buffer("vec_scale", vec_scale, persistent=False)
        self.out_dim = n_channels * (n_channels + 1) // 2

    def _covariance(self, x_patches: torch.Tensor) -> torch.Tensor:
        """x_patches: [..., C, L] -> [..., C, C] regularized SPD covariance."""
        L = x_patches.shape[-1]
        cov = torch.matmul(x_patches, x_patches.transpose(-1, -2)) / max(L - 1, 1)
        eye = torch.eye(self.n_channels, device=x_patches.device, dtype=x_patches.dtype)
        return cov + self.eps * eye  # diagonal shrinkage for numerical SPD safety

    def _log_map(self, cov: torch.Tensor) -> torch.Tensor:
        """Regularized matrix log with condition-number clamping.

        Eigenvalues are floored at ``eps`` and additionally raised so the
        ratio lambda_max / lambda_min never exceeds ``max_cond`` -- this
        keeps Logm well-conditioned for covariance estimates from short,
        possibly near-rank-deficient patches, without discarding the
        dominant eigenstructure.
        """
        evals, evecs = torch.linalg.eigh(cov)  # ascending order, cov: [..., C, C]
        evals = torch.clamp(evals, min=self.eps)
        lambda_max = evals.max(dim=-1, keepdim=True).values
        lambda_min_allowed = lambda_max / self.max_cond
        evals = torch.maximum(evals, lambda_min_allowed)
        log_evals = torch.log(evals)
        return evecs @ torch.diag_embed(log_evals) @ evecs.transpose(-1, -2)

    def _vectorize(self, sym_mat: torch.Tensor) -> torch.Tensor:
        """sym_mat: [..., C, C] -> [..., C*(C+1)/2], Frobenius-preserving."""
        vec = sym_mat[..., self.triu_row, self.triu_col]
        return vec * self.vec_scale.to(dtype=sym_mat.dtype)

    def forward(self, x_patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_patches: [B, N, C, L] -> (tangent_vectors [B, N, D], frechet_mean_logm [B, C, C])."""
        cov = self._covariance(x_patches)          # [B, N, C, C]
        log_cov = self._log_map(cov)                # [B, N, C, C]
        frechet_mean_logm = log_cov.mean(dim=1)      # [B, C, C], Frechet mean in log-Euclidean metric
        centered = log_cov - frechet_mean_logm.unsqueeze(1)  # tangent vectors at the Frechet mean
        tangent_vectors = self._vectorize(centered)  # [B, N, D]
        return tangent_vectors, frechet_mean_logm


# --------------------------------------------------------------------------
# Transformer building block (MC-Dropout throughout)
# --------------------------------------------------------------------------


class _TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block with MC Dropout in place of nn.Dropout."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_p: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.drop_attn = MCDropout(dropout_p)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            MCDropout(dropout_p),
            nn.Linear(d_ff, d_model),
        )
        self.drop_ff = MCDropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_attn(attn_out)
        h = self.norm2(x)
        x = x + self.drop_ff(self.ff(h))
        return x


class _AttentionPool(nn.Module):
    """Learned-query attention pooling over the patch/token dimension."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        return pooled.squeeze(1)  # [B, d_model]


class _ScaleBranch(nn.Module):
    """One temporal-resolution branch: patchify -> embed -> transformer -> pool.

    Optionally fuses Riemannian tangent-space tokens (aligned patch-for-
    patch) into the same-scale raw-signal token embedding.
    """

    def __init__(
        self,
        n_channels: int,
        patch_len: int,
        seq_len: int,
        tangent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout_p: float,
        use_spatial: bool,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by patch_len={patch_len}")
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.use_spatial = use_spatial

        self.raw_embed = nn.Linear(n_channels * patch_len, d_model)
        if use_spatial:
            self.spatial_embed = nn.Linear(tangent_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.input_dropout = MCDropout(dropout_p)
        self.blocks = nn.ModuleList(
            [_TransformerEncoderBlock(d_model, n_heads, d_ff, dropout_p) for _ in range(n_layers)]
        )
        self.pool = _AttentionPool(d_model, n_heads)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> [B, N, C, patch_len]."""
        B, C, T = x.shape
        return x[:, :, : self.n_patches * self.patch_len].reshape(B, C, self.n_patches, self.patch_len).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, tangent: torch.Tensor = None) -> torch.Tensor:
        patches = self.patchify(x)  # [B, N, C, patch_len]
        B, N, C, L = patches.shape
        tokens = self.raw_embed(patches.reshape(B, N, C * L))
        if self.use_spatial:
            if tangent is None:
                raise ValueError("use_spatial=True but no tangent vectors were provided")
            tokens = tokens + self.spatial_embed(tangent)
        tokens = tokens + self.pos_embed
        tokens = self.input_dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return self.pool(tokens)  # [B, d_model]


# --------------------------------------------------------------------------
# Module 2: multi-scale PatchTST-style encoder
# --------------------------------------------------------------------------


class PatchTSTEncoder(nn.Module):
    """Multi-scale temporal transformer fusing raw and Riemannian-tangent tokens.

    Each entry in ``patch_lens`` defines an independent temporal
    resolution ("scale"); ``spatial_scale_idx`` selects which scale's
    patches also receive spatial covariance features via
    :class:`RiemannianTangentProjector`. Per-scale summaries (attention-
    pooled over patches) are concatenated and linearly fused into a single
    representation. MC Dropout (see :class:`MCDropout`) is used throughout
    every branch.
    """

    def __init__(
        self,
        n_channels: int = DEFAULT_N_CHANNELS,
        seq_len: int = DEFAULT_SEQ_LEN,
        patch_lens: Sequence[int] = (128, 256),
        spatial_scale_idx: int = 0,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        if not (0 <= spatial_scale_idx < len(patch_lens)):
            raise ValueError("spatial_scale_idx must index into patch_lens")

        self.patch_lens = tuple(patch_lens)
        self.spatial_scale_idx = spatial_scale_idx
        self.spatial_patch_len = self.patch_lens[spatial_scale_idx]

        self.tangent_projector = RiemannianTangentProjector(n_channels)
        self.branches = nn.ModuleList(
            [
                _ScaleBranch(
                    n_channels=n_channels,
                    patch_len=pl,
                    seq_len=seq_len,
                    tangent_dim=self.tangent_projector.out_dim,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    dropout_p=dropout_p,
                    use_spatial=(i == spatial_scale_idx),
                )
                for i, pl in enumerate(self.patch_lens)
            ]
        )
        fused_dim = d_model * len(self.patch_lens)
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            MCDropout(dropout_p),
        )
        self.out_dim = fused_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] raw multichannel EEG epoch -> [B, out_dim] fused representation."""
        n_patches_spatial = x.shape[-1] // self.spatial_patch_len
        spatial_patches = (
            x[:, :, : n_patches_spatial * self.spatial_patch_len]
            .reshape(x.shape[0], x.shape[1], n_patches_spatial, self.spatial_patch_len)
            .permute(0, 2, 1, 3)
        )  # [B, N, C, L]
        tangent, _frechet_mean_logm = self.tangent_projector(spatial_patches)

        scale_summaries = []
        for i, branch in enumerate(self.branches):
            tangent_in = tangent if i == self.spatial_scale_idx else None
            scale_summaries.append(branch(x, tangent_in))
        fused = torch.cat(scale_summaries, dim=-1)
        return self.fuse(fused)


# --------------------------------------------------------------------------
# Module 3: full backbone
# --------------------------------------------------------------------------


class NeuroLensBackbone(nn.Module):
    """Encoder -> unit-norm 128-d latent state + binary seizure-risk logit."""

    def __init__(
        self,
        n_channels: int = DEFAULT_N_CHANNELS,
        seq_len: int = DEFAULT_SEQ_LEN,
        patch_lens: Sequence[int] = (128, 256),
        spatial_scale_idx: int = 0,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout_p: float = 0.3,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.encoder = PatchTSTEncoder(
            n_channels=n_channels,
            seq_len=seq_len,
            patch_lens=patch_lens,
            spatial_scale_idx=spatial_scale_idx,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout_p=dropout_p,
        )
        enc_dim = self.encoder.out_dim
        self.latent_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
            MCDropout(dropout_p),
            nn.Linear(enc_dim, latent_dim),
        )
        self.classifier = nn.Sequential(
            MCDropout(dropout_p),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            MCDropout(dropout_p),
            nn.Linear(latent_dim // 2, 1),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, n_channels, seq_len] -> {'latent_vector': [B, latent_dim], 'seizure_logits': [B]}."""
        h = self.encoder(x)
        z = F.normalize(self.latent_head(h), p=2, dim=-1)
        logits = self.classifier(z).squeeze(-1)
        return {"latent_vector": z, "seizure_logits": logits}

    @torch.no_grad()
    def get_mc_prediction(self, x: torch.Tensor, num_samples: int = 30) -> Dict[str, torch.Tensor]:
        """Monte Carlo Dropout inference: epistemic mean probability and predictive entropy.

        Because every dropout layer in this model is :class:`MCDropout`
        (stochastic regardless of train/eval mode), ``num_samples``
        independent stochastic forward passes are obtained cheaply by
        replicating the batch along a new leading dimension and running a
        single forward call -- each replica draws independent dropout
        masks -- rather than looping in Python.

        Returns:
            mean_prob: [B] MC-averaged seizure probability (epistemic mean).
            predictive_entropy: [B] binary entropy of mean_prob (total
                predictive uncertainty).
            mc_std: [B] std. dev. of the per-sample MC probabilities (a
                purer epistemic-uncertainty signal, since it vanishes when
                the model is confident and consistent across dropout masks).
        """
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        was_training = self.training
        self.eval()  # freeze any train/eval-sensitive layers (e.g. future BatchNorm); MCDropout ignores this
        try:
            B = x.shape[0]
            x_rep = x.repeat(num_samples, *([1] * (x.dim() - 1)))  # [num_samples * B, ...]
            logits = self.forward(x_rep)["seizure_logits"].view(num_samples, B)
            probs = torch.sigmoid(logits)
        finally:
            self.train(was_training)

        mean_prob = probs.mean(dim=0)
        mc_std = probs.std(dim=0)
        eps = 1e-8
        predictive_entropy = -(
            mean_prob * torch.log(mean_prob + eps) + (1.0 - mean_prob) * torch.log(1.0 - mean_prob + eps)
        )
        return {"mean_prob": mean_prob, "predictive_entropy": predictive_entropy, "mc_std": mc_std}
