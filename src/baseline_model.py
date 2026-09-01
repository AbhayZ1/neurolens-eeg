"""Standard-deep-learning baseline for NeuroLens ablations.

A plain 1D-adapted ResNet-18 (He et al., 2016 -- every Conv2d/BatchNorm2d/
pooling op replaced with its 1D counterpart, in_channels=18 in place of
RGB's 3) trained with vanilla `torch.nn.BCEWithLogitsLoss`. No Riemannian
tangent-space features, no multi-scale patch transformer, no Supervised
Contrastive pretraining, no trajectory/FAISS retrieval, and no
always-on Monte Carlo Dropout -- deliberately so: this exists to isolate
how much of NeuroLens's performance comes from its dynamical-trajectory-
aware representation versus a standard, strong, off-the-shelf time-series
classifier trained directly for the task.

Input contract matches dataset.py exactly: [B, 18, 1280] (18-channel
bipolar montage, 5-second epochs at 256 Hz), so this drops into the same
`get_loso_splits` DataLoaders NeuroLensBackbone uses.

Training note: pair this model with a plain `torch.nn.BCEWithLogitsLoss()`
directly at the call site -- there is deliberately no custom loss wrapper
here (unlike DynamicalSupConLoss, this baseline has nothing but a single
BCE term to wrap).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

# Matches dataset.py: N_CHANNELS=18, N_SAMPLES=1280.
DEFAULT_IN_CHANNELS = 18
DEFAULT_SEQ_LEN = 1280


class BasicBlock1D(nn.Module):
    """The standard ResNet "basic block" (He et al., 2016, Fig. 2 left),
    with every 2D op replaced by its 1D counterpart: two 3-wide
    convolutions with a residual (identity or projection) shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """ResNet-18 adapted for 1D multichannel time series.

    Structure (matches the canonical 18-weight-layer count exactly):
    a 7-wide stride-2 stem conv + stride-2 max-pool, then 4 stages of 2
    BasicBlocks each (4*2*2 = 16 conv layers) at widths
    [base_width, 2x, 4x, 8x], global average pooling, and a single linear
    classification head. Default `layers=(2,2,2,2)` and `base_width=64`
    reproduce ResNet-18's proportions; both are configurable for a
    "lightweight" variant (e.g. `base_width=32`) if the full-size model is
    more capacity than a single-patient LOSO fold's training set warrants.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        layers: Sequence[int] = (2, 2, 2, 2),
        base_width: int = 64,
        dropout_p: float = 0.3,
        num_classes: int = 1,
    ):
        super().__init__()
        if len(layers) != 4:
            raise ValueError(f"layers must have exactly 4 stages, got {len(layers)}")

        self._in_planes = base_width
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(base_width, layers[0], stride=1)
        self.layer2 = self._make_stage(base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_stage(base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_stage(base_width * 8, layers[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_p)  # standard dropout: OFF at eval() -- intentional, see module docstring
        self.fc = nn.Linear(base_width * 8, num_classes)

        self.feature_dim = base_width * 8
        self._init_weights()

    def _make_stage(self, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self._in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self._in_planes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        blocks = [BasicBlock1D(self._in_planes, out_channels, stride=stride, downsample=downsample)]
        self._in_planes = out_channels
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock1D(self._in_planes, out_channels))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        # Zero-init the last BN in each residual block (Goyal et al., 2017):
        # every residual branch starts as an identity map, which measurably
        # improves early-training stability for ResNets.
        for m in self.modules():
            if isinstance(m, BasicBlock1D):
                nn.init.constant_(m.bn2.weight, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, T] -> pooled feature vector [B, feature_dim], pre-dropout/pre-fc."""
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.global_pool(h).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, T] -> raw logits [B] (apply sigmoid for seizure probability).

        T must be large enough to survive five stride-2 downsamplings
        (stem conv + stem pool + 3 of the 4 stages) without collapsing to
        zero length; T=1280 (dataset.py's 5 s @ 256 Hz epoch) safely
        reduces to 40 before global pooling.
        """
        features = self.extract_features(x)
        features = self.dropout(features)
        return self.fc(features).squeeze(-1)


def resnet18_1d(
    in_channels: int = DEFAULT_IN_CHANNELS, dropout_p: float = 0.3, num_classes: int = 1
) -> ResNet1D:
    """Factory matching the canonical ResNet-18 depth/width (base_width=64, layers=(2,2,2,2))."""
    return ResNet1D(in_channels=in_channels, layers=(2, 2, 2, 2), base_width=64, dropout_p=dropout_p, num_classes=num_classes)


def resnet18_1d_lightweight(
    in_channels: int = DEFAULT_IN_CHANNELS, dropout_p: float = 0.3, num_classes: int = 1
) -> ResNet1D:
    """A narrower variant (base_width=32, ~1/4 the parameters) for small
    single-patient LOSO training sets, where the full-width ResNet-18 may
    have far more capacity than the data can constrain."""
    return ResNet1D(in_channels=in_channels, layers=(2, 2, 2, 2), base_width=32, dropout_p=dropout_p, num_classes=num_classes)


if __name__ == "__main__":
    # Smoke test: confirm the architecture runs end-to-end on dataset.py's
    # exact tensor contract and that gradients flow.
    model = resnet18_1d()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"resnet18_1d: {n_params:,} parameters")

    x = torch.randn(4, DEFAULT_IN_CHANNELS, DEFAULT_SEQ_LEN) * 20e-6
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])

    logits = model(x)
    print(f"logits shape: {tuple(logits.shape)}")
    assert logits.shape == (4,)

    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"loss={loss.item():.4f}, params with grad: {n_with_grad}/{len(list(model.parameters()))}")
    assert n_with_grad == len(list(model.parameters()))
    print("OK")
