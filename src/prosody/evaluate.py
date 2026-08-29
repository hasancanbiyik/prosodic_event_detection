"""Evaluation utilities and trivial baselines.

The single most important thing this module does is compute **trivial
baselines**, so any reported model F1 can be put in context.

For a binary task with positive rate ``p``, the always-predict-positive F1 is
``2p / (1 + p)``. If a model scores below that, it's worse than predicting
every frame positive — which the original README's "85.3% prominence F1"
landed only ~5 points above on the *easier* sequence-level task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class TaskMetrics:
    task: str
    f1: float
    precision: float
    recall: float
    auc: float | None
    positive_rate: float
    trivial_majority_f1: float
    n_frames: int
    confusion: tuple[int, int, int, int]  # tn, fp, fn, tp


def trivial_baselines(y_true: np.ndarray) -> dict[str, float]:
    """Return F1 of always-zero and always-one baselines."""
    p = float(np.mean(y_true)) if len(y_true) else 0.0
    always_one_f1 = (2 * p) / (1 + p) if p > 0 else 0.0
    always_zero_f1 = 0.0  # by convention F1 with zero TP is 0
    return {
        "positive_rate": p,
        "always_one_f1": always_one_f1,
        "always_zero_f1": always_zero_f1,
        "majority_class_f1": max(always_one_f1, always_zero_f1),
    }


def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    *,
    task: str,
) -> TaskMetrics:
    p = float(np.mean(y_true)) if len(y_true) else 0.0
    triv = (2 * p) / (1 + p) if p > 0 else 0.0
    auc: float | None = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            auc = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    return TaskMetrics(
        task=task,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        auc=auc,
        positive_rate=p,
        trivial_majority_f1=triv,
        n_frames=len(y_true),
        confusion=(tn, fp, fn, tp),
    )


def evaluate_frame_level(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: np.ndarray | None = None,
    task: str = "prominence",
) -> TaskMetrics:
    """Evaluate per-frame predictions."""
    return _binary_metrics(y_true, y_pred, y_proba, task=task)


def evaluate_sequence_level(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: np.ndarray | None = None,
    task: str = "prominence",
) -> TaskMetrics:
    """Evaluate per-window predictions (same math, different label space)."""
    return _binary_metrics(y_true, y_pred, y_proba, task=task)


def format_metrics(m: TaskMetrics) -> str:
    """Pretty single-line summary."""
    auc = f"AUC={m.auc:.3f}" if m.auc is not None else "AUC=n/a"
    return (
        f"{m.task:>10s}: F1={m.f1:.3f}  P={m.precision:.3f}  R={m.recall:.3f}  "
        f"{auc}  | pos_rate={m.positive_rate:.3f}  "
        f"trivial_F1={m.trivial_majority_f1:.3f}  N={m.n_frames}"
    )
