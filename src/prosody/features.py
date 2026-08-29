"""Acoustic feature extraction.

We extract a 16-dimensional frame-level feature vector at 10 ms hop:

* F0 (1) — librosa ``yin`` (50–400 Hz)
* RMS energy (1)
* Spectral centroid (1)
* MFCCs 1–13 (13)

This matches the original notebook's feature set so downstream comparisons are
on the same input space.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16_000
FRAME_SHIFT_MS = 10.0
FRAME_LENGTH_MS = 25.0
N_MFCC = 13
N_FEATURES = 3 + N_MFCC  # F0 + Energy + SpectralCentroid + 13 MFCCs

FEATURE_NAMES: tuple[str, ...] = (
    "F0",
    "RMS_Energy",
    "Spectral_Centroid",
    *(f"MFCC_{i + 1}" for i in range(N_MFCC)),
)


def _frame_lengths(sr: int) -> tuple[int, int]:
    frame_length = int(sr * FRAME_LENGTH_MS / 1000.0)
    hop_length = int(sr * FRAME_SHIFT_MS / 1000.0)
    return frame_length, hop_length


def extract_features(
    audio_path: str | Path,
    *,
    target_sr: int = TARGET_SR,
    fmin: float = 50.0,
    fmax: float = 400.0,
    nan_policy: str = "median",
) -> np.ndarray:
    """Load audio and return a ``(n_frames, 16)`` float32 feature matrix.

    Parameters
    ----------
    audio_path
        Path to a ``.wav`` file. The file is resampled to ``target_sr``.
    target_sr
        Target sample rate (default 16 kHz).
    fmin, fmax
        F0 search range for ``librosa.yin``.
    nan_policy
        How to handle non-finite values produced by ``yin`` for unvoiced frames.
        ``"median"`` (default) imputes per-column with the median; ``"zero"``
        replaces with 0; ``"raise"`` raises ``ValueError``.

    Returns
    -------
    np.ndarray
        Shape ``(n_frames, N_FEATURES)``, dtype ``float32``.
    """
    y, sr = librosa.load(str(audio_path), sr=target_sr)
    frame_length, hop_length = _frame_lengths(sr)

    f0 = librosa.yin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length
    )
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length)

    # Align lengths defensively (librosa primitives can disagree by 1 frame).
    n_frames = min(len(f0), len(energy), len(centroid), mfcc.shape[1])
    feats = np.vstack(
        [
            f0[:n_frames].reshape(1, -1),
            energy[:n_frames].reshape(1, -1),
            centroid[:n_frames].reshape(1, -1),
            mfcc[:, :n_frames],
        ]
    ).T.astype(np.float32)

    feats = _sanitize(feats, policy=nan_policy)
    return feats


def _sanitize(feats: np.ndarray, *, policy: str) -> np.ndarray:
    """Replace NaN/Inf in-place per column."""
    bad = ~np.isfinite(feats)
    if not bad.any():
        return feats
    if policy == "raise":
        raise ValueError(f"Found {int(bad.sum())} non-finite values in features")
    if policy == "zero":
        feats[bad] = 0.0
        return feats
    # median policy
    for c in range(feats.shape[1]):
        col = feats[:, c]
        col_bad = ~np.isfinite(col)
        if col_bad.any():
            good = col[~col_bad]
            fill = float(np.median(good)) if good.size else 0.0
            col[col_bad] = fill
    return feats


def n_frames_for(audio_path: str | Path, *, target_sr: int = TARGET_SR) -> int:
    """Estimate number of feature frames for a wav without loading it fully."""
    info = sf.info(str(audio_path))
    duration_s = info.frames / info.samplerate
    return int(duration_s * 1000.0 / FRAME_SHIFT_MS)
