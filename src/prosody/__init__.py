"""Prosodic event detection on the Boston University Radio News Corpus.

The package is split into modules so individual pieces can be used without
pulling in heavy optional dependencies (e.g. ``torch``):

* :mod:`prosody.labels`   — ToBI parsing, frame/sequence label generation
  *(no torch required)*
* :mod:`prosody.features` — librosa-based feature extraction *(no torch)*
* :mod:`prosody.data`     — corpus loading, speaker-aware splits *(no torch)*
* :mod:`prosody.evaluate` — metrics, trivial baselines *(no torch)*
* :mod:`prosody.models`   — Conv1D / BiLSTM models *(requires torch)*
* :mod:`prosody.train`    — classical + neural training loops
  *(neural pieces require torch; classical pieces don't)*
* :mod:`prosody.cli`      — ``prosody`` command line entry point

Re-exports here are lazy: importing :mod:`prosody.models` is what triggers the
torch import, not just ``import prosody``.
"""

from __future__ import annotations

__version__ = "0.2.0"

# Eagerly re-export torch-free utilities for ergonomics.
# Classical training has no torch dependency.
from prosody._train_classical import (  # noqa: F401
    ClassicalResult,
    train_frame_level_classical,
    train_sequence_level_classical,
)
from prosody.data import CorpusFile, load_corpus, speaker_split  # noqa: F401
from prosody.evaluate import (  # noqa: F401
    evaluate_frame_level,
    evaluate_sequence_level,
    trivial_baselines,
)
from prosody.features import FRAME_SHIFT_MS, N_FEATURES, extract_features  # noqa: F401
from prosody.labels import (  # noqa: F401
    ToBIEvent,
    ToBIParser,
    aggregate_to_sequence_labels,
    frame_labels_from_events,
    sequence_windows,
)


def __getattr__(name: str):
    """Lazy access to torch-requiring objects.

    >>> from prosody import ProminenceCNN  # imports torch only on this line
    """
    if name in {
        "FrameClassifier",
        "ProminenceBiLSTM",
        "ProminenceCNN",
        "SequenceClassifier",
        "ModelConfig",
    }:
        from prosody import models as _m

        return getattr(_m, name)
    if name in {
        "train_frame_level_neural",
        "train_sequence_level_neural",
    }:
        from prosody import train as _t

        return getattr(_t, name)
    raise AttributeError(f"module 'prosody' has no attribute {name!r}")


__all__ = [
    "CorpusFile",
    "load_corpus",
    "speaker_split",
    "FRAME_SHIFT_MS",
    "N_FEATURES",
    "extract_features",
    "ToBIEvent",
    "ToBIParser",
    "aggregate_to_sequence_labels",
    "frame_labels_from_events",
    "sequence_windows",
    "evaluate_frame_level",
    "evaluate_sequence_level",
    "trivial_baselines",
    # lazy:
    "FrameClassifier",
    "SequenceClassifier",
    "ProminenceCNN",
    "ProminenceBiLSTM",
    "ModelConfig",
    "train_frame_level_classical",
    "train_frame_level_neural",
    "train_sequence_level_classical",
    "train_sequence_level_neural",
]
