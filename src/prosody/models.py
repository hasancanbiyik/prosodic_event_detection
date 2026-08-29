"""Neural model definitions.

Two task heads are supported:

* **Frame-level (sequence-to-sequence)** — produces a per-frame logit. Used
  when comparing against the classical ML frame-level baseline.
* **Sequence-level (pooled)** — produces one logit per fixed window. Used when
  comparing against classical ML aggregated over windows.

Both tasks are multi-task (prominence + boundary) sharing a backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    in_channels: int = 16
    hidden_channels: int = 64
    rnn_hidden: int = 64
    rnn_layers: int = 2
    dropout: float = 0.2


# ---------------------------------------------------------------------------
# Frame-level (sequence-to-sequence) backbones
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 5, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return self.net(x)


class FrameClassifier(nn.Module):
    """Per-frame classifier with two task heads (prominence, boundary).

    Input  : ``(B, T, F)`` — features (B batch, T frames, F=16 channels).
    Output : ``(prominence_logits, boundary_logits)`` each of shape ``(B, T)``.

    Architecture: three stacked dilated 1D conv blocks (receptive field 29 frames
    = 290 ms), then a 1×1 head. This is intentionally small — the goal is to be
    a fair comparison point against a frame-level logistic regression / random
    forest, not to chase state of the art.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        h = cfg.hidden_channels
        self.backbone = nn.Sequential(
            _ConvBlock(cfg.in_channels, h, k=5, dilation=1, dropout=cfg.dropout),
            _ConvBlock(h, h, k=5, dilation=2, dropout=cfg.dropout),
            _ConvBlock(h, h, k=5, dilation=4, dropout=cfg.dropout),
        )
        self.prom_head = nn.Conv1d(h, 1, kernel_size=1)
        self.bound_head = nn.Conv1d(h, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.backbone(x)
        prom = self.prom_head(h).squeeze(1)
        bound = self.bound_head(h).squeeze(1)
        return prom, bound


# ---------------------------------------------------------------------------
# Sequence-level (pooled) backbones — replicate the original notebook behavior
# ---------------------------------------------------------------------------


class ProminenceCNN(nn.Module):
    """Sequence-level 1D CNN: one logit per (B, T, F) window.

    This is a compact reimplementation for the original notebook's easier
    500 ms window task; it is not an exact replica of the archived architecture.
    We pool over time with adaptive average pooling, then use one linear layer
    per task head.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        h = cfg.hidden_channels
        self.backbone = nn.Sequential(
            _ConvBlock(cfg.in_channels, h, k=5, dropout=cfg.dropout),
            _ConvBlock(h, h, k=5, dropout=cfg.dropout),
            _ConvBlock(h, h * 2, k=5, dropout=cfg.dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.prom_head = nn.Linear(h * 2, 1)
        self.bound_head = nn.Linear(h * 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)  # (B, F, T)
        h = self.backbone(x)
        h = self.pool(h).squeeze(-1)  # (B, C)
        return self.prom_head(h).squeeze(-1), self.bound_head(h).squeeze(-1)


class ProminenceBiLSTM(nn.Module):
    """Sequence-level Bi-LSTM: one logit per (B, T, F) window."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.rnn = nn.LSTM(
            input_size=cfg.in_channels,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
        )
        self.prom_head = nn.Linear(cfg.rnn_hidden * 2, 1)
        self.bound_head = nn.Linear(cfg.rnn_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.rnn(x)  # (B, T, 2H)
        pooled = out.mean(dim=1)  # mean pool over time
        return self.prom_head(pooled).squeeze(-1), self.bound_head(pooled).squeeze(-1)


class SequenceClassifier(nn.Module):
    """Wrapper to pick CNN or BiLSTM by name."""

    def __init__(self, kind: str = "cnn", cfg: ModelConfig | None = None):
        super().__init__()
        kind = kind.lower()
        if kind == "cnn":
            self.net = ProminenceCNN(cfg)
        elif kind in {"bilstm", "rnn", "lstm"}:
            self.net = ProminenceBiLSTM(cfg)
        else:
            raise ValueError(f"unknown sequence model kind: {kind!r}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)
