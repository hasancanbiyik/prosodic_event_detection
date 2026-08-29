"""Classical-ML training (no torch dependency).

Kept separate from :mod:`prosody._train_neural` so users can run frame-level /
sequence-level baselines without installing PyTorch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from prosody.data import CorpusFile
from prosody.evaluate import TaskMetrics, evaluate_frame_level, evaluate_sequence_level
from prosody.labels import sequence_windows

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers shared with the neural side
# ---------------------------------------------------------------------------


def stack_frame_level(files: Sequence[CorpusFile]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate features and labels frame-wise. Returns ``(X, y_prom, y_bound)``."""
    X = np.vstack([f.features for f in files]).astype(np.float32)
    y_prom = np.concatenate([f.prominence_labels for f in files]).astype(np.int64)
    y_bound = np.concatenate([f.boundary_labels for f in files]).astype(np.int64)
    return X, y_prom, y_bound


def build_sequence_level(
    files: Sequence[CorpusFile],
    *,
    window: int = 50,
    hop: int | None = None,
    prom_threshold: float = 0.15,
    bound_threshold: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice each file into fixed windows.

    Returns ``(X_seq, y_prom_seq, y_bound_seq, y_prom_frame, y_bound_frame)``.
    """
    X_chunks: list[np.ndarray] = []
    yp_seq: list[int] = []
    yb_seq: list[int] = []
    yp_frame: list[np.ndarray] = []
    yb_frame: list[np.ndarray] = []
    for f in files:
        for s, e in sequence_windows(f.n_frames, window=window, hop=hop):
            X_chunks.append(f.features[s:e])
            prom_chunk = f.prominence_labels[s:e]
            bound_chunk = f.boundary_labels[s:e]
            yp_frame.append(prom_chunk.astype(np.int64))
            yb_frame.append(bound_chunk.astype(np.int64))
            yp_seq.append(int(prom_chunk.mean() >= prom_threshold))
            yb_seq.append(int(bound_chunk.mean() >= bound_threshold))
    if not X_chunks:
        n_feat = files[0].features.shape[1] if files else 16
        return (
            np.zeros((0, window, n_feat), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, window), dtype=np.int64),
            np.zeros((0, window), dtype=np.int64),
        )
    X_seq = np.stack(X_chunks).astype(np.float32)
    return (
        X_seq,
        np.asarray(yp_seq, dtype=np.int64),
        np.asarray(yb_seq, dtype=np.int64),
        np.stack(yp_frame),
        np.stack(yb_frame),
    )


def build_aggregated_features(X_seq: np.ndarray) -> np.ndarray:
    """Mean / std / min / max pool features within each window.

    Input ``(n_windows, T, F)`` -> output ``(n_windows, 4F)``.
    """
    if X_seq.size == 0:
        return X_seq.reshape(0, X_seq.shape[-1] * 4)
    feats = np.concatenate(
        [
            X_seq.mean(axis=1),
            X_seq.std(axis=1),
            X_seq.min(axis=1),
            X_seq.max(axis=1),
        ],
        axis=1,
    )
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Result dataclass + model zoo
# ---------------------------------------------------------------------------


@dataclass
class ClassicalResult:
    task: str
    model_name: str
    val: TaskMetrics
    test: TaskMetrics
    scaler: StandardScaler
    estimator: object
    importance: np.ndarray | None = None


def _classical_models(seed: int = 42) -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def _fit_one(
    name, est, Xtr_s, y_train, Xva_s, y_val, Xte_s, y_test, task, eval_fn
) -> ClassicalResult:
    est.fit(Xtr_s, y_train)
    proba_val = est.predict_proba(Xva_s)[:, 1] if hasattr(est, "predict_proba") else None
    proba_test = est.predict_proba(Xte_s)[:, 1] if hasattr(est, "predict_proba") else None
    pred_val = est.predict(Xva_s)
    pred_test = est.predict(Xte_s)
    return ClassicalResult(
        task=task,
        model_name=name,
        val=eval_fn(y_val, pred_val, y_proba=proba_val, task=task),
        test=eval_fn(y_test, pred_test, y_proba=proba_test, task=task),
        scaler=None,  # caller fills in
        estimator=est,
        importance=getattr(est, "feature_importances_", None),
    )


# ---------------------------------------------------------------------------
# Cell 1: classical, frame-level
# ---------------------------------------------------------------------------


def train_frame_level_classical(
    train_files: Sequence[CorpusFile],
    val_files: Sequence[CorpusFile],
    test_files: Sequence[CorpusFile],
    *,
    seed: int = 42,
) -> dict[str, dict[str, ClassicalResult]]:
    """Classical ML where every 10 ms frame is one example."""
    X_train, ypt, ybt = stack_frame_level(train_files)
    X_val, ypv, ybv = stack_frame_level(val_files)
    X_test, ypte, ybte = stack_frame_level(test_files)

    scaler = StandardScaler().fit(X_train)
    # scikit-learn's optimizers are more numerically stable in float64. The
    # source feature matrices stay float32 to keep corpus caches compact.
    Xtr = scaler.transform(X_train).astype(np.float64)
    Xva = scaler.transform(X_val).astype(np.float64)
    Xte = scaler.transform(X_test).astype(np.float64)

    out: dict[str, dict[str, ClassicalResult]] = {"prominence": {}, "boundary": {}}
    for task, y_train, y_val, y_test in [
        ("prominence", ypt, ypv, ypte),
        ("boundary", ybt, ybv, ybte),
    ]:
        for name, est in _classical_models(seed).items():
            r = _fit_one(
                name, est, Xtr, y_train, Xva, y_val, Xte, y_test, task, evaluate_frame_level
            )
            r.scaler = scaler
            out[task][name] = r
            logger.info(
                "[frame/classical] %s %s val_F1=%.3f test_F1=%.3f triv=%.3f",
                task,
                name,
                r.val.f1,
                r.test.f1,
                r.test.trivial_majority_f1,
            )
    return out


# ---------------------------------------------------------------------------
# Cell 2: classical, sequence-level (the comparison the original lacked)
# ---------------------------------------------------------------------------


def train_sequence_level_classical(
    train_files: Sequence[CorpusFile],
    val_files: Sequence[CorpusFile],
    test_files: Sequence[CorpusFile],
    *,
    window: int = 50,
    hop: int | None = None,
    prom_threshold: float = 0.15,
    bound_threshold: float = 0.10,
    seed: int = 42,
) -> dict[str, dict[str, ClassicalResult]]:
    """Aggregate features per 50-frame window, run classical ML on the same task as the NN."""
    X_train, yp_train, yb_train, _, _ = build_sequence_level(
        train_files,
        window=window,
        hop=hop,
        prom_threshold=prom_threshold,
        bound_threshold=bound_threshold,
    )
    X_val, yp_val, yb_val, _, _ = build_sequence_level(
        val_files,
        window=window,
        hop=hop,
        prom_threshold=prom_threshold,
        bound_threshold=bound_threshold,
    )
    X_test, yp_test, yb_test, _, _ = build_sequence_level(
        test_files,
        window=window,
        hop=hop,
        prom_threshold=prom_threshold,
        bound_threshold=bound_threshold,
    )

    Xtr = build_aggregated_features(X_train)
    Xva = build_aggregated_features(X_val)
    Xte = build_aggregated_features(X_test)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float64)
    Xva_s = scaler.transform(Xva).astype(np.float64)
    Xte_s = scaler.transform(Xte).astype(np.float64)

    out: dict[str, dict[str, ClassicalResult]] = {"prominence": {}, "boundary": {}}
    for task, y_train, y_val, y_test in [
        ("prominence", yp_train, yp_val, yp_test),
        ("boundary", yb_train, yb_val, yb_test),
    ]:
        for name, est in _classical_models(seed).items():
            r = _fit_one(
                name,
                est,
                Xtr_s,
                y_train,
                Xva_s,
                y_val,
                Xte_s,
                y_test,
                task,
                evaluate_sequence_level,
            )
            r.scaler = scaler
            out[task][name] = r
            logger.info(
                "[seq/classical] %s %s val_F1=%.3f test_F1=%.3f triv=%.3f",
                task,
                name,
                r.val.f1,
                r.test.f1,
                r.test.trivial_majority_f1,
            )
    return out
