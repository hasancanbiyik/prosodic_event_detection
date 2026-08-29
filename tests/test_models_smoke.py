"""Smoke tests for model forward passes — catches shape errors before training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from prosody.data import CorpusFile
from prosody.models import FrameClassifier, ProminenceBiLSTM, ProminenceCNN, SequenceClassifier
from prosody.train import _eval_frame_files


def test_frame_classifier_shapes():
    model = FrameClassifier()
    x = torch.from_numpy(np.random.randn(4, 50, 16).astype(np.float32))
    prom, bound = model(x)
    assert prom.shape == (4, 50)
    assert bound.shape == (4, 50)


def test_sequence_cnn_shapes():
    model = ProminenceCNN()
    x = torch.from_numpy(np.random.randn(4, 50, 16).astype(np.float32))
    prom, bound = model(x)
    assert prom.shape == (4,)
    assert bound.shape == (4,)


def test_sequence_bilstm_shapes():
    model = ProminenceBiLSTM()
    x = torch.from_numpy(np.random.randn(4, 50, 16).astype(np.float32))
    prom, bound = model(x)
    assert prom.shape == (4,)
    assert bound.shape == (4,)


def test_sequence_classifier_kind_dispatch():
    model = SequenceClassifier(kind="bilstm")
    x = torch.from_numpy(np.random.randn(2, 50, 16).astype(np.float32))
    p, b = model(x)
    assert p.shape == b.shape == (2,)


def test_one_step_train_runs():
    """Sanity: a single backward pass shouldn't NaN out."""
    model = FrameClassifier()
    x = torch.from_numpy(np.random.randn(4, 50, 16).astype(np.float32))
    yp = torch.randint(0, 2, (4, 50)).float()
    yb = torch.randint(0, 2, (4, 50)).float()
    p, b = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        p, yp
    ) + torch.nn.functional.binary_cross_entropy_with_logits(b, yb)
    loss.backward()
    assert torch.isfinite(loss)


def test_frame_evaluation_counts_each_corpus_frame_once():
    rng = np.random.default_rng(42)
    files = []
    for index, n_frames in enumerate((17, 23)):
        features = rng.normal(size=(n_frames, 16)).astype(np.float32)
        files.append(
            CorpusFile(
                file_id=f"f1a-{index}",
                speaker="f1a",
                audio_path=Path(f"unused-{index}.wav"),
                features=features,
                prominence_labels=np.zeros(n_frames, dtype=np.int8),
                boundary_labels=np.zeros(n_frames, dtype=np.int8),
            )
        )

    scaler = StandardScaler().fit(np.vstack([item.features for item in files]))
    prominence, boundary = _eval_frame_files(FrameClassifier(), files, scaler, torch.device("cpu"))

    assert prominence.n_frames == 40
    assert boundary.n_frames == 40
