"""Run the full classical-ML half of the comparison matrix on the real corpus.

This is the part of the experiment that doesn't need ``torch``, so it can be
verified inside a CPU-only sandbox before launching the full ``run-all`` job.

Output: writes ``artifacts/classical_results.json`` and prints a summary table
to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

from prosody._train_classical import (
    train_frame_level_classical,
    train_sequence_level_classical,
)
from prosody.data import load_corpus, speaker_split, summarize
from prosody.evaluate import format_metrics


def _load_corpus(args):
    cache = Path(args.cache)
    use_cache = not args.no_cache and args.max_files is None
    if cache.exists() and use_cache:
        print(f"Loading cached corpus from {cache}", file=sys.stderr)
        with cache.open("rb") as handle:
            return pickle.load(handle)
    print("Building corpus from audio...", file=sys.stderr)
    corpus = load_corpus(args.data_root, max_files=args.max_files, progress=False)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as handle:
            pickle.dump(corpus, handle)
        print(f"Cached corpus to {cache}", file=sys.stderr)
    return corpus


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="AutoRPT_Data")
    p.add_argument("--cache", default="artifacts/corpus.pkl")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--val-speaker", default="m2b")
    p.add_argument("--test-speaker", default="f3a")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/classical_results.json")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    corpus = _load_corpus(args)
    stats = summarize(corpus)
    print(json.dumps(asdict(stats), indent=2), file=sys.stderr)

    sp = speaker_split(corpus, val_speakers=[args.val_speaker], test_speakers=[args.test_speaker])
    print(f"\nSplit: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}", file=sys.stderr)

    print("\n=== Frame-level classical ML ===")
    cf = train_frame_level_classical(sp.train, sp.val, sp.test, seed=args.seed)
    for task, models in cf.items():
        for name, r in models.items():
            print(f"  {format_metrics(r.test):<110s}  ← {name} [{task}]")

    print("\n=== Sequence-level classical ML ===")
    cs = train_sequence_level_classical(sp.train, sp.val, sp.test, seed=args.seed)
    for task, models in cs.items():
        for name, r in models.items():
            print(f"  {format_metrics(r.test):<110s}  ← {name} [{task}]")

    out = {
        "stats": asdict(stats),
        "frame_classical": _serialize(cf),
        "sequence_classical": _serialize(cs),
        "split": {
            "train_speakers": sorted({c.speaker for c in sp.train}),
            "val_speakers": sorted({c.speaker for c in sp.val}),
            "test_speakers": sorted({c.speaker for c in sp.test}),
        },
        "config": {
            "seed": args.seed,
            "window_frames": 50,
            "hop_frames": 25,
            "prominence_threshold": 0.15,
            "boundary_threshold": 0.10,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


def _metric_to_dict(m) -> dict:
    d = asdict(m)
    d["confusion"] = list(d["confusion"])
    return d


def _serialize(results) -> dict:
    out = {}
    for task, models in results.items():
        out[task] = {}
        for name, r in models.items():
            out[task][name] = {"val": _metric_to_dict(r.val), "test": _metric_to_dict(r.test)}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
