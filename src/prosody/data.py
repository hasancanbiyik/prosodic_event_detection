"""Data loading: scan AutoRPT_Data/, build per-file records, speaker-aware splits.

The corpus on disk is laid out as::

    AutoRPT_Data/
        f1a/
            j/  f1ajrlp1.wav  f1ajrlp1.ton  f1ajrlp1.TextGrid
                ...
            p/  ...
            r/  ...
            t/  ...
        f2b/  ...
        m1b/  ...
        ...

The first three characters of the file stem are the speaker ID (e.g. ``f1a``).
We use this to perform **speaker-aware splits**: any given speaker appears in
    exactly one of train / val / test. This is what was missing from the original
    notebook's file-order split.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from prosody.features import FRAME_SHIFT_MS, extract_features
from prosody.labels import ToBIParser, find_ton_path, frame_labels_from_events

logger = logging.getLogger(__name__)


@dataclass
class CorpusFile:
    """One processed audio file: features + frame-level labels."""

    file_id: str
    speaker: str
    audio_path: Path
    features: np.ndarray  # (n_frames, n_features) float32
    prominence_labels: np.ndarray  # (n_frames,) int8
    boundary_labels: np.ndarray  # (n_frames,) int8
    n_prominence_events: int = 0
    n_boundary_events: int = 0
    duration_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.features.shape[0])


def _speaker_from_stem(stem: str) -> str:
    """``f1ajrlp1`` -> ``f1a``."""
    return stem[:3]


def discover_audio_files(data_root: str | Path) -> list[Path]:
    """Return all ``.wav`` files under ``data_root`` that have a sibling ``.ton``."""
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    wavs = sorted(data_root.rglob("*.wav"))
    keep: list[Path] = []
    skipped: list[Path] = []
    for w in wavs:
        if w.with_suffix(".ton").exists():
            keep.append(w)
        else:
            skipped.append(w)
    if skipped:
        logger.warning(
            "Skipped %d wav file(s) without matching .ton: %s",
            len(skipped),
            ", ".join(str(p.relative_to(data_root)) for p in skipped[:5]),
        )
    return keep


def load_corpus(
    data_root: str | Path,
    *,
    max_files: int | None = None,
    tolerance_ms: float = 50.0,
    parser: ToBIParser | None = None,
    progress: bool = True,
) -> list[CorpusFile]:
    """Build :class:`CorpusFile` records for every audio file with a ``.ton``.

    Parameters
    ----------
    data_root
        Path to ``AutoRPT_Data/``.
    max_files
        Cap for quick smoke runs.
    tolerance_ms
        Half-width of the positive window around each ToBI event.
    parser
        Optional pre-configured :class:`ToBIParser`.
    progress
        Show tqdm progress bar.
    """
    parser = parser or ToBIParser()
    audio_files = discover_audio_files(data_root)
    if max_files is not None:
        audio_files = audio_files[:max_files]

    records: list[CorpusFile] = []
    iterator: Iterable[Path] = (
        tqdm(audio_files, desc="Processing audio") if progress else audio_files
    )

    for wav in iterator:
        try:
            ton = find_ton_path(wav)
            prom_times, bound_times = parser.parse_events_by_type(ton)
            feats = extract_features(wav)
            n_frames = feats.shape[0]
            prom = frame_labels_from_events(
                prom_times, n_frames, frame_shift_ms=FRAME_SHIFT_MS, tolerance_ms=tolerance_ms
            )
            bound = frame_labels_from_events(
                bound_times,
                n_frames,
                frame_shift_ms=FRAME_SHIFT_MS,
                tolerance_ms=tolerance_ms,
            )
            duration_s = n_frames * FRAME_SHIFT_MS / 1000.0
            records.append(
                CorpusFile(
                    file_id=wav.stem,
                    speaker=_speaker_from_stem(wav.stem),
                    audio_path=wav,
                    features=feats,
                    prominence_labels=prom,
                    boundary_labels=bound,
                    n_prominence_events=len(prom_times),
                    n_boundary_events=len(bound_times),
                    duration_s=duration_s,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", wav.name, exc)
    return records


@dataclass
class CorpusStats:
    n_files: int
    n_speakers: int
    total_minutes: float
    total_frames: int
    prominence_rate: float
    boundary_rate: float
    per_speaker: dict[str, int]


def summarize(corpus: Sequence[CorpusFile]) -> CorpusStats:
    """Compute summary statistics across a list of :class:`CorpusFile`."""
    total_frames = sum(c.n_frames for c in corpus)
    total_minutes = sum(c.duration_s for c in corpus) / 60.0
    prom_pos = sum(int(c.prominence_labels.sum()) for c in corpus)
    bound_pos = sum(int(c.boundary_labels.sum()) for c in corpus)
    speakers: dict[str, int] = {}
    for c in corpus:
        speakers[c.speaker] = speakers.get(c.speaker, 0) + 1
    return CorpusStats(
        n_files=len(corpus),
        n_speakers=len(speakers),
        total_minutes=total_minutes,
        total_frames=total_frames,
        prominence_rate=(prom_pos / total_frames) if total_frames else 0.0,
        boundary_rate=(bound_pos / total_frames) if total_frames else 0.0,
        per_speaker=speakers,
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass
class Split:
    train: list[CorpusFile]
    val: list[CorpusFile]
    test: list[CorpusFile]


def speaker_split(
    corpus: Sequence[CorpusFile],
    *,
    val_speakers: Sequence[str] = ("m2b",),
    test_speakers: Sequence[str] = ("f3a",),
) -> Split:
    """Split files so each speaker appears in exactly one of train/val/test.

    Defaults: validate on ``m2b`` (one male), test on ``f3a`` (one female).
    Train gets the remaining 4 speakers.

    Why this matters
    ----------------
    The original notebook sliced the corpus by file order. With only 6 speakers,
    that did not guarantee disjoint talkers across splits. Speaker-aware splits
    give a more realistic estimate of generalization to a *new* talker.
    """
    val_set = set(val_speakers)
    test_set = set(test_speakers)
    overlap = val_set & test_set
    if overlap:
        raise ValueError(f"val and test cannot share speakers: {overlap}")

    train, val, test = [], [], []
    for c in corpus:
        if c.speaker in test_set:
            test.append(c)
        elif c.speaker in val_set:
            val.append(c)
        else:
            train.append(c)
    return Split(train=train, val=val, test=test)


def file_split_random(
    corpus: Sequence[CorpusFile],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Split:
    """Random file-level split for experiments that do not require speaker isolation."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(corpus))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return Split(
        train=[corpus[i] for i in train_idx],
        val=[corpus[i] for i in val_idx],
        test=[corpus[i] for i in test_idx],
    )
