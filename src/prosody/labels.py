"""ToBI label parsing and frame-level / sequence-level label generation.

The Boston University Radio News Corpus ships ToBI annotations as ``.ton`` files
in a simple tabular format::

    signal f1ajrlp1
    type 0
    color 115
    comment created using xlabel ...
    font ...
    separator ;
    nfields 1
    #
        0.284780  115 H*
        0.513538  115 L-L%
        0.975722  115 H*
        ...

Each non-header row is ``<time_seconds> <color_code> <label>``.

We follow the standard ToBI conventions for prosodic event detection:

* **Prominence** (a.k.a. *pitch accents*): labels containing ``*`` such as
  ``H*``, ``L*``, ``L+H*``, ``L*+H``, ``H+!H*``, ``!H*``, even uncertain ``*?``.
* **Boundary** (default: *intonational phrase boundary*): labels ending in
  ``%`` such as ``L-L%``, ``H-L%``, ``!H-L%``, ``L-H%``. Callers can opt into
  intermediate-phrase labels ending in ``-`` (for example ``L-`` or ``H-``).

We deliberately exclude:

* ``HiF0`` markers (these tag the high-F0 of the *previous* accent, not a
  separate event)
* break-index-only labels and silent boundaries that have no tone content

These choices are documented in :class:`ToBIParser` so a reviewer can see
exactly what got mapped to what.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

EventType = Literal["prominence", "boundary", "other"]
BoundaryScope = Literal["intonational", "all"]


@dataclass(frozen=True)
class ToBIEvent:
    """A single ToBI annotation parsed from a ``.ton`` file."""

    time_s: float
    raw_label: str
    event_type: EventType


# Pre-compiled patterns. Order matters: prominence is checked first because some
# labels (like ``L+H*``) contain both ``+`` and ``*``; we want them as accents.
_PROMINENCE_RE = re.compile(r"\*")
# A label is an intonational phrase boundary iff it ends in ``%`` (e.g. L-L%, H-L%).
_INTONATIONAL_RE = re.compile(r"%$")
# A label is an intermediate phrase boundary iff it ends in ``-`` (e.g. L-, !H-).
_INTERMEDIATE_RE = re.compile(r"-$")
# HiF0 / NoF0 / pseudo-tones we ignore.
_IGNORE_TOKENS = {"HiF0", "NoF0", "BREAK", ".", "<", ">"}


def _normalize_label(label: str) -> str:
    """Take only the first whitespace token of the label field.

    The annotation files sometimes append free-text comments after the label
    (e.g. ``L-L%      ; low amplitude breath, not visible``). Without
    normalization those rows would be misclassified as "other" because the
    boundary regex expects ``%`` or ``-`` at end-of-string.
    """
    return label.strip().split(None, 1)[0] if label.strip() else ""


def _classify_label(label: str, *, boundary_scope: BoundaryScope = "intonational") -> EventType:
    """Map a raw ToBI label string to a high-level event type.

    Parameters
    ----------
    label
        Raw label text from a ``.ton`` row. Free-text comments are stripped.
    boundary_scope
        ``"intonational"`` (default, matches BURNC literature) → only labels
        ending in ``%`` count as boundaries. ``"all"`` → also include
        intermediate-phrase labels ending in ``-``.

    >>> _classify_label("H*")
    'prominence'
    >>> _classify_label("L+H*")
    'prominence'
    >>> _classify_label("!H*")
    'prominence'
    >>> _classify_label("*?")
    'prominence'
    >>> _classify_label("L-L%")
    'boundary'
    >>> _classify_label("!H-L%")
    'boundary'
    >>> _classify_label("H-")
    'other'
    >>> _classify_label("H-", boundary_scope="all")
    'boundary'
    >>> _classify_label("HiF0")
    'other'
    """
    label = _normalize_label(label)
    if not label or label in _IGNORE_TOKENS:
        return "other"
    if _PROMINENCE_RE.search(label):
        return "prominence"
    if _INTONATIONAL_RE.search(label):
        return "boundary"
    if boundary_scope == "all" and _INTERMEDIATE_RE.search(label):
        return "boundary"
    return "other"


class ToBIParser:
    """Parser for AutoRPT ``.ton`` files.

    Notes
    -----
    The header in a ``.ton`` file is terminated by a single ``#`` line. We skip
    everything up to and including that line, then parse data rows.

    Each data row has at least three whitespace-separated fields:
    ``time, color, label[, label, ...]``. We join everything after the color
    field as the raw label, then classify it.
    """

    def __init__(
        self,
        *,
        include_uncertain: bool = True,
        boundary_scope: BoundaryScope = "intonational",
    ) -> None:
        self.include_uncertain = include_uncertain
        self.boundary_scope = boundary_scope

    def parse_file(self, path: str | Path) -> list[ToBIEvent]:
        """Parse a single ``.ton`` file into a list of :class:`ToBIEvent`."""
        path = Path(path)
        with path.open("r", encoding="latin-1") as f:
            lines = f.readlines()

        # Locate end of header (line that is just '#').
        header_end = next((i for i, ln in enumerate(lines) if ln.strip() == "#"), -1)
        if header_end == -1:
            raise ValueError(f"{path}: no '#' header terminator found")

        events: list[ToBIEvent] = []
        for raw in lines[header_end + 1 :]:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            label = parts[2].strip()
            if not self.include_uncertain and _normalize_label(label).endswith("?"):
                continue
            kind = _classify_label(label, boundary_scope=self.boundary_scope)
            events.append(ToBIEvent(time_s=t, raw_label=label, event_type=kind))
        return events

    def parse_events_by_type(self, path: str | Path) -> tuple[list[float], list[float]]:
        """Convenience: return ``(prominence_times, boundary_times)`` in seconds."""
        events = self.parse_file(path)
        prom = [e.time_s for e in events if e.event_type == "prominence"]
        bound = [e.time_s for e in events if e.event_type == "boundary"]
        return prom, bound


def frame_labels_from_events(
    event_times_s: Sequence[float],
    n_frames: int,
    *,
    frame_shift_ms: float = 10.0,
    tolerance_ms: float = 50.0,
) -> np.ndarray:
    """Convert a list of event timestamps into a 0/1 frame-level label vector.

    Each event marks all frames within ``±tolerance_ms`` as positive.

    Parameters
    ----------
    event_times_s
        Event timestamps in seconds (e.g. accent peak times from a ``.ton`` file).
    n_frames
        Total number of frames to produce.
    frame_shift_ms
        Frame hop in milliseconds. Default 10 ms (matches AutoRPT convention).
    tolerance_ms
        Half-width of the positive window around each event. Default 50 ms (i.e.
        a 100 ms window centered on the timestamp). This is the same tolerance
        used in the original notebook.

    Returns
    -------
    np.ndarray
        ``int8`` array of shape ``(n_frames,)``.
    """
    if frame_shift_ms <= 0:
        raise ValueError("frame_shift_ms must be positive")
    labels = np.zeros(n_frames, dtype=np.int8)
    if not event_times_s:
        return labels
    tol_frames = int(round(tolerance_ms / frame_shift_ms))
    for t in event_times_s:
        if t is None or np.isnan(t):
            continue
        center = int(round(t * 1000.0 / frame_shift_ms))
        start = max(0, center - tol_frames)
        end = min(n_frames, center + tol_frames + 1)
        if end > start:
            labels[start:end] = 1
    return labels


def sequence_windows(
    n_frames: int,
    *,
    window: int = 50,
    hop: int | None = None,
    drop_last: bool = True,
) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` frame indices for fixed-length sliding windows.

    Parameters
    ----------
    n_frames
        Number of available frames.
    window
        Window length in frames. Default 50 frames (= 500 ms at 10 ms shift).
    hop
        Step size. Defaults to ``window // 2`` (50% overlap, matching the
        original notebook).
    drop_last
        If True, drop trailing windows that don't fit. Otherwise the last
        window is right-anchored at ``n_frames``.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if hop is None:
        hop = max(1, window // 2)
    last_start = n_frames - window
    if last_start < 0:
        return
    for start in range(0, last_start + 1, hop):
        yield start, start + window
    if not drop_last and (last_start % hop) != 0:
        yield last_start, n_frames


def aggregate_to_sequence_labels(
    frame_labels: np.ndarray,
    *,
    window: int = 50,
    hop: int | None = None,
    threshold: float = 0.15,
) -> np.ndarray:
    """Aggregate a frame-level 0/1 vector to one binary label per window.

    A window is positive iff the fraction of positive frames within it is at
    least ``threshold``. The notebook used 0.15 for prominence and 0.10 for
    boundary; we expose the threshold so callers can tune per task.
    """
    out: list[int] = []
    for s, e in sequence_windows(len(frame_labels), window=window, hop=hop):
        out.append(int(frame_labels[s:e].mean() >= threshold))
    return np.asarray(out, dtype=np.int8)


def find_ton_path(audio_path: str | Path) -> Path:
    """Given ``foo.wav``, return the sibling ``foo.ton`` (raise if missing)."""
    audio_path = Path(audio_path)
    ton = audio_path.with_suffix(".ton")
    if not ton.exists():
        raise FileNotFoundError(f"No .ton annotation alongside {audio_path}")
    return ton


def iter_ton_events(path: str | Path) -> Iterable[ToBIEvent]:
    """Convenience iterator using the default :class:`ToBIParser` config."""
    yield from ToBIParser().parse_file(path)
