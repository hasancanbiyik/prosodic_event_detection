"""Smoke tests for the feature extraction pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from prosody.features import N_FEATURES, extract_features


@pytest.fixture
def synthetic_wav(tmp_path: Path) -> Path:
    sr = 16_000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    # 200 Hz sine + a small amount of noise so spectral features aren't degenerate
    y = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.01 * np.random.default_rng(0).standard_normal(len(t))
    path = tmp_path / "tone.wav"
    sf.write(path, y, sr, subtype="PCM_16")
    return path


def test_extract_features_shape(synthetic_wav: Path):
    feats = extract_features(synthetic_wav)
    # 2 s @ 10 ms hop ≈ 200 frames (depending on librosa's centering, ±a few)
    assert feats.ndim == 2
    assert feats.shape[1] == N_FEATURES == 16
    assert feats.shape[0] >= 195
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_extract_features_handles_unvoiced(synthetic_wav: Path):
    # NaN policy 'median' should leave no NaNs even for silent inputs
    sr = 16_000
    silent = np.zeros(int(sr * 1.0), dtype=np.float32)
    path = synthetic_wav.parent / "silent.wav"
    sf.write(path, silent, sr, subtype="PCM_16")
    feats = extract_features(path, nan_policy="median")
    assert np.isfinite(feats).all()
