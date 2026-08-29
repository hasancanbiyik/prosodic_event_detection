"""Training routines for the four cells of the comparison matrix.

                       │ frame-level (per 10 ms)  │ sequence-level (per 500 ms)
    ───────────────────┼──────────────────────────┼─────────────────────────────
    classical ML       │ train_frame_level_       │ train_sequence_level_
                       │     classical            │     classical
    ───────────────────┼──────────────────────────┼─────────────────────────────
    neural network     │ train_frame_level_       │ train_sequence_level_
                       │     neural (Conv1D s2s)  │     neural (CNN / BiLSTM)

The whole point is that *both rows* and *both columns* are reported, so the
headline comparison is honest.

The classical-ML training paths live in :mod:`prosody._train_classical` and do
**not** import torch — you can run frame-level / sequence-level baselines
without installing PyTorch. The neural paths in this module obviously need it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from prosody._train_classical import (  # noqa: F401  (re-exports)
    ClassicalResult,
    build_aggregated_features,
    build_sequence_level,
    stack_frame_level,
    train_frame_level_classical,
    train_sequence_level_classical,
)
from prosody.data import CorpusFile
from prosody.evaluate import TaskMetrics, evaluate_frame_level, evaluate_sequence_level
from prosody.models import FrameClassifier, ModelConfig, SequenceClassifier

logger = logging.getLogger(__name__)


def _device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(max(1.0, neg / pos))


@dataclass
class NeuralResult:
    task_kind: str  # "frame" or "sequence"
    model_name: str
    val_metrics: dict[str, TaskMetrics]
    test_metrics: dict[str, TaskMetrics]
    history: list[dict] = field(default_factory=list)
    checkpoint_path: Path | None = None


def _normalize_X_seq(X_seq: np.ndarray, scaler):
    """Flatten -> fit/transform -> reshape back. Returns ``(X, scaler)``."""
    from sklearn.preprocessing import StandardScaler

    if X_seq.size == 0:
        return X_seq, scaler or StandardScaler()
    n, t, f = X_seq.shape
    flat = X_seq.reshape(-1, f)
    if scaler is None:
        scaler = StandardScaler().fit(flat)
    flat_s = scaler.transform(flat)
    return flat_s.reshape(n, t, f).astype(np.float32), scaler


# ---------------------------------------------------------------------------
# Frame-level NN (s2s)
# ---------------------------------------------------------------------------


class _FrameWindowDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, yp_frame: np.ndarray, yb_frame: np.ndarray):
        self.X = torch.from_numpy(X_seq).float()
        self.yp = torch.from_numpy(yp_frame).float()
        self.yb = torch.from_numpy(yb_frame).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": self.X[i], "yp": self.yp[i], "yb": self.yb[i]}


def _eval_frame_files(model, files: Sequence[CorpusFile], scaler, device):
    """Evaluate every original corpus frame exactly once.

    Training uses overlapping windows to provide local context, but flattening
    those windows at evaluation time double-counts most frames. A fully
    convolutional model can score a complete file directly, preserving the
    original 10 ms task and making its metrics comparable with the classical
    frame-level baselines.
    """
    model.eval()
    yp_true, yb_true, yp_pred, yb_pred, yp_prob, yb_prob = [], [], [], [], [], []
    with torch.no_grad():
        for corpus_file in files:
            features = scaler.transform(corpus_file.features).astype(np.float32)
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            lp, lb = model(x)
            pp = torch.sigmoid(lp).cpu().numpy().reshape(-1)
            pb = torch.sigmoid(lb).cpu().numpy().reshape(-1)
            yp_prob.append(pp)
            yb_prob.append(pb)
            yp_pred.append((pp >= 0.5).astype(np.int64))
            yb_pred.append((pb >= 0.5).astype(np.int64))
            yp_true.append(corpus_file.prominence_labels.astype(np.int64))
            yb_true.append(corpus_file.boundary_labels.astype(np.int64))
    return (
        evaluate_frame_level(
            np.concatenate(yp_true),
            np.concatenate(yp_pred),
            y_proba=np.concatenate(yp_prob),
            task="prominence",
        ),
        evaluate_frame_level(
            np.concatenate(yb_true),
            np.concatenate(yb_pred),
            y_proba=np.concatenate(yb_prob),
            task="boundary",
        ),
    )


def train_frame_level_neural(
    train_files: Sequence[CorpusFile],
    val_files: Sequence[CorpusFile],
    test_files: Sequence[CorpusFile],
    *,
    window: int = 50,
    hop: int | None = None,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    cfg: ModelConfig | None = None,
    seed: int = 42,
    checkpoint_dir: str | Path = "artifacts",
) -> NeuralResult:
    """Frame-level Conv1D s2s model. Per-frame BCE loss on every frame in the window."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, _, _, yp_tr, yb_tr = build_sequence_level(train_files, window=window, hop=hop)

    Xtr, scaler = _normalize_X_seq(Xtr, None)

    train_ds = _FrameWindowDataset(Xtr, yp_tr, yb_tr)

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    device = _device()
    model = FrameClassifier(cfg).to(device)
    pw_prom = _pos_weight(yp_tr.reshape(-1)).to(device)
    pw_bound = _pos_weight(yb_tr.reshape(-1)).to(device)
    loss_prom = nn.BCEWithLogitsLoss(pos_weight=pw_prom)
    loss_bound = nn.BCEWithLogitsLoss(pos_weight=pw_bound)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    best = -1.0
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "frame_level_cnn.pt"

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(dl_train, desc=f"frame-ep{ep}", leave=False):
            x = batch["x"].to(device)
            yp = batch["yp"].to(device)
            yb = batch["yb"].to(device)
            opt.zero_grad()
            lp, lb = model(x)
            loss = loss_prom(lp, yp) + loss_bound(lb, yb)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / max(1, len(train_ds))
        prom_m, bound_m = _eval_frame_files(model, val_files, scaler, device)
        avg = (prom_m.f1 + bound_m.f1) / 2.0
        history.append(
            dict(epoch=ep, train_loss=train_loss, val_prom_f1=prom_m.f1, val_bound_f1=bound_m.f1)
        )
        logger.info(
            "[frame/nn] epoch %d loss=%.4f val_prom_F1=%.3f val_bound_F1=%.3f",
            ep,
            train_loss,
            prom_m.f1,
            bound_m.f1,
        )
        if avg > best:
            best = avg
            torch.save(
                {"model_state": model.state_dict(), "scaler": scaler, "cfg": cfg},
                ckpt_path,
            )

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    val_prom, val_bound = _eval_frame_files(model, val_files, scaler, device)
    test_prom, test_bound = _eval_frame_files(model, test_files, scaler, device)
    return NeuralResult(
        task_kind="frame",
        model_name="FrameClassifierConv1D",
        val_metrics={"prominence": val_prom, "boundary": val_bound},
        test_metrics={"prominence": test_prom, "boundary": test_bound},
        history=history,
        checkpoint_path=ckpt_path,
    )


# ---------------------------------------------------------------------------
# Sequence-level NN (the original task)
# ---------------------------------------------------------------------------


class _SequenceWindowDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, yp_seq: np.ndarray, yb_seq: np.ndarray):
        self.X = torch.from_numpy(X_seq).float()
        self.yp = torch.from_numpy(yp_seq).float()
        self.yb = torch.from_numpy(yb_seq).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": self.X[i], "yp": self.yp[i], "yb": self.yb[i]}


def _eval_sequence_loader(model, loader, device):
    model.eval()
    yp_true, yb_true, yp_pred, yb_pred, yp_prob, yb_prob = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lp, lb = model(x)
            pp = torch.sigmoid(lp).cpu().numpy()
            pb = torch.sigmoid(lb).cpu().numpy()
            yp_prob.append(pp)
            yb_prob.append(pb)
            yp_pred.append((pp >= 0.5).astype(np.int64))
            yb_pred.append((pb >= 0.5).astype(np.int64))
            yp_true.append(batch["yp"].numpy().astype(np.int64))
            yb_true.append(batch["yb"].numpy().astype(np.int64))
    return (
        evaluate_sequence_level(
            np.concatenate(yp_true),
            np.concatenate(yp_pred),
            y_proba=np.concatenate(yp_prob),
            task="prominence",
        ),
        evaluate_sequence_level(
            np.concatenate(yb_true),
            np.concatenate(yb_pred),
            y_proba=np.concatenate(yb_prob),
            task="boundary",
        ),
    )


def train_sequence_level_neural(
    train_files: Sequence[CorpusFile],
    val_files: Sequence[CorpusFile],
    test_files: Sequence[CorpusFile],
    *,
    kind: str = "cnn",
    window: int = 50,
    hop: int | None = None,
    prom_threshold: float = 0.15,
    bound_threshold: float = 0.10,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    cfg: ModelConfig | None = None,
    seed: int = 42,
    checkpoint_dir: str | Path = "artifacts",
) -> NeuralResult:
    """Sequence-level CNN or BiLSTM. One label per 50-frame window."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, yp_tr, yb_tr, _, _ = build_sequence_level(
        train_files,
        window=window,
        hop=hop,
        prom_threshold=prom_threshold,
        bound_threshold=bound_threshold,
    )
    Xva, yp_va, yb_va, _, _ = build_sequence_level(
        val_files,
        window=window,
        hop=hop,
        prom_threshold=prom_threshold,
        bound_threshold=bound_threshold,
    )
    Xte, yp_te, yb_te, _, _ = build_sequence_level(
        test_files,
        window=window,
        hop=hop,
        prom_threshold=prom_threshold,
        bound_threshold=bound_threshold,
    )

    Xtr, scaler = _normalize_X_seq(Xtr, None)
    Xva, _ = _normalize_X_seq(Xva, scaler)
    Xte, _ = _normalize_X_seq(Xte, scaler)

    dl_train = DataLoader(
        _SequenceWindowDataset(Xtr, yp_tr, yb_tr), batch_size=batch_size, shuffle=True
    )
    dl_val = DataLoader(_SequenceWindowDataset(Xva, yp_va, yb_va), batch_size=batch_size)
    dl_test = DataLoader(_SequenceWindowDataset(Xte, yp_te, yb_te), batch_size=batch_size)

    device = _device()
    model = SequenceClassifier(kind=kind, cfg=cfg).to(device)
    pw_prom = _pos_weight(yp_tr).to(device)
    pw_bound = _pos_weight(yb_tr).to(device)
    loss_prom = nn.BCEWithLogitsLoss(pos_weight=pw_prom)
    loss_bound = nn.BCEWithLogitsLoss(pos_weight=pw_bound)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    best = -1.0
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"sequence_level_{kind.lower()}.pt"

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(dl_train, desc=f"seq-{kind}-ep{ep}", leave=False):
            x = batch["x"].to(device)
            yp = batch["yp"].to(device)
            yb = batch["yb"].to(device)
            opt.zero_grad()
            lp, lb = model(x)
            loss = loss_prom(lp, yp) + loss_bound(lb, yb)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / max(1, len(dl_train.dataset))
        prom_m, bound_m = _eval_sequence_loader(model, dl_val, device)
        avg = (prom_m.f1 + bound_m.f1) / 2.0
        history.append(
            dict(epoch=ep, train_loss=train_loss, val_prom_f1=prom_m.f1, val_bound_f1=bound_m.f1)
        )
        logger.info(
            "[seq/nn:%s] epoch %d loss=%.4f val_prom_F1=%.3f val_bound_F1=%.3f",
            kind,
            ep,
            train_loss,
            prom_m.f1,
            bound_m.f1,
        )
        if avg > best:
            best = avg
            torch.save(
                {"model_state": model.state_dict(), "scaler": scaler, "kind": kind, "cfg": cfg},
                ckpt_path,
            )

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    val_prom, val_bound = _eval_sequence_loader(model, dl_val, device)
    test_prom, test_bound = _eval_sequence_loader(model, dl_test, device)
    return NeuralResult(
        task_kind="sequence",
        model_name=f"SequenceClassifier-{kind}",
        val_metrics={"prominence": val_prom, "boundary": val_bound},
        test_metrics={"prominence": test_prom, "boundary": test_bound},
        history=history,
        checkpoint_path=ckpt_path,
    )
