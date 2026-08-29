"""Tests for ToBI parsing and frame-label generation.

These tests document the label classification rules that the rest of the
project depends on, so a future change to the parser can't silently shift the
labels under our trained models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prosody.labels import (
    ToBIParser,
    _classify_label,  # type: ignore[attr-defined]
    aggregate_to_sequence_labels,
    frame_labels_from_events,
    sequence_windows,
)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("H*", "prominence"),
        ("L*", "prominence"),
        ("L+H*", "prominence"),
        ("L*+H", "prominence"),
        ("!H*", "prominence"),
        ("H+!H*", "prominence"),
        ("*?", "prominence"),
        # Default boundary scope is intonational ('%' suffix only).
        ("L-L%", "boundary"),
        ("H-L%", "boundary"),
        ("!H-L%", "boundary"),
        ("L-H%", "boundary"),
        # Intermediate phrase (just '-') does NOT count by default.
        ("L-", "other"),
        ("H-", "other"),
        ("HiF0", "other"),
        ("NoF0", "other"),
        ("", "other"),
        (" ", "other"),
    ],
)
def test_classify_label_canonical(label: str, expected: str):
    assert _classify_label(label) == expected


@pytest.mark.parametrize(
    "label, expected",
    [
        ("L-", "boundary"),
        ("H-", "boundary"),
        ("!H-", "boundary"),
        ("L-L%", "boundary"),
    ],
)
def test_classify_label_all_scope(label: str, expected: str):
    assert _classify_label(label, boundary_scope="all") == expected


def test_classify_label_strips_inline_comments():
    # Real BURNC rows look like:  "    1.234  115 L-L%      ; low amplitude breath"
    raw = "L-L%      ; low amplitude breath, not visible"
    assert _classify_label(raw) == "boundary"


def test_parser_handles_actual_ton_file(tmp_path: Path):
    sample = """signal f1ajrlp1
type 0
color 115
comment created using xlabel
font foo
separator ;
nfields 1
#
    0.284780  115 H*
    0.284780  115 HiF0
    0.513538  115 L-L%
    0.975722  115 H*
    1.349205  115 !H*
    2.110175  115 L+H*
    3.011201  115 *?
    3.342667  115 L-L%
    4.395082  115 H*
"""
    path = tmp_path / "f1ajrlp1.ton"
    path.write_text(sample)

    prom, bound = ToBIParser().parse_events_by_type(path)

    assert len(prom) == 6  # H*, H*, !H*, L+H*, *?, H*
    assert len(bound) == 2  # two L-L% (intonational; default scope)
    # HiF0 is filtered (has no '*' and no '%' suffix)
    assert all(t > 0 for t in prom + bound)

    # Switching to 'all' scope would not change this fixture (no '-' boundaries)
    prom_all, bound_all = ToBIParser(boundary_scope="all").parse_events_by_type(path)
    assert (prom_all, bound_all) == (prom, bound)


def test_frame_labels_basic():
    # Single event at t=1.0s, ±50 ms tolerance, 10 ms shift, 200 frames.
    labels = frame_labels_from_events([1.0], n_frames=200, tolerance_ms=50.0)
    # frame 100 should be the center; ±5 frames around it should be 1
    assert labels[100] == 1
    assert labels[95] == 1
    assert labels[105] == 1
    assert labels[94] == 0
    assert labels[106] == 0
    assert labels.sum() == 11  # 5 + 1 + 5


def test_frame_labels_no_events_returns_zeros():
    labels = frame_labels_from_events([], n_frames=100)
    assert labels.shape == (100,)
    assert labels.sum() == 0


def test_frame_labels_clamps_at_boundaries():
    # Event near t=0 should not produce negative indices
    labels = frame_labels_from_events([0.005], n_frames=20, tolerance_ms=50.0)
    assert labels[0] == 1
    # Event past the audio end gets ignored
    labels2 = frame_labels_from_events([10.0], n_frames=20, tolerance_ms=50.0)
    assert labels2.sum() == 0


def test_sequence_windows_50pct_overlap():
    windows = list(sequence_windows(150, window=50, hop=25))
    # starts: 0, 25, 50, 75, 100 (last_start=100). Length = 5.
    assert [s for s, _ in windows] == [0, 25, 50, 75, 100]
    assert all((e - s) == 50 for s, e in windows)


def test_aggregate_to_sequence_labels_threshold():
    # 0.2 positive rate over 50 frames -> 10 positives
    fl = np.zeros(50, dtype=np.int8)
    fl[:10] = 1
    out_15 = aggregate_to_sequence_labels(fl, window=50, hop=50, threshold=0.15)
    out_25 = aggregate_to_sequence_labels(fl, window=50, hop=50, threshold=0.25)
    assert out_15.tolist() == [1]  # 0.20 >= 0.15
    assert out_25.tolist() == [0]  # 0.20 < 0.25
