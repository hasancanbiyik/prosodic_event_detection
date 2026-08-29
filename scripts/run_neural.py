"""Run the neural half of the comparison matrix on the cached corpus.

Run after ``scripts/run_classical_only.py`` (or any preprocessing that produces
``artifacts/corpus.pkl``). This trains:

* Frame-level Conv1D s2s (one logit per 10 ms frame)
* Sequence-level CNN (one logit per 500 ms window)
* Sequence-level BiLSTM (one logit per 500 ms window)

The sequence models use the same task definition as the original notebook but
are compact reimplementations, not exact replicas of its CNN and attention-BiGRU.

and writes ``artifacts/neural_results.json`` plus per-model checkpoints in
``artifacts/`` (each with a unique filename — fixes the bug where the original
notebook's CNN got overwritten by the RNN).

Requires ``torch`` (CPU is fine; GPU/MPS is auto-detected).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

from prosody.data import load_corpus, speaker_split, summarize
from prosody.evaluate import format_metrics
from prosody.train import train_frame_level_neural, train_sequence_level_neural


def _load_corpus(args):
    cache = Path(args.cache)
    use_cache = not args.no_cache and args.max_files is None
    if cache.exists() and use_cache:
        print(f"Loading cached corpus from {cache}", file=sys.stderr)
        with cache.open("rb") as f:
            return pickle.load(f)
    print("Building corpus from audio...", file=sys.stderr)
    corpus = load_corpus(args.data_root, max_files=args.max_files, progress=False)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump(corpus, f)
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
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/neural_results.json")
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
    print(
        f"Split: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}",
        file=sys.stderr,
    )

    results = {
        "stats": asdict(stats),
        "split": {
            "train_speakers": sorted({c.speaker for c in sp.train}),
            "val_speakers": sorted({c.speaker for c in sp.val}),
            "test_speakers": sorted({c.speaker for c in sp.test}),
        },
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "window_frames": 50,
            "hop_frames": 25,
            "prominence_threshold": 0.15,
            "boundary_threshold": 0.10,
        },
    }

    print("\n=== Frame-level Conv1D s2s ===")
    nf = train_frame_level_neural(
        sp.train,
        sp.val,
        sp.test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    results["frame_neural"] = _serialize(nf)
    for task, m in nf.test_metrics.items():
        print(f"  [{task:>10s}] {nf.model_name:<24s} {format_metrics(m)}")

    print("\n=== Sequence-level CNN ===")
    ncnn = train_sequence_level_neural(
        sp.train,
        sp.val,
        sp.test,
        kind="cnn",
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    results["sequence_neural_cnn"] = _serialize(ncnn)
    for task, m in ncnn.test_metrics.items():
        print(f"  [{task:>10s}] {ncnn.model_name:<24s} {format_metrics(m)}")

    print("\n=== Sequence-level BiLSTM ===")
    nbil = train_sequence_level_neural(
        sp.train,
        sp.val,
        sp.test,
        kind="bilstm",
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    results["sequence_neural_bilstm"] = _serialize(nbil)
    for task, m in nbil.test_metrics.items():
        print(f"  [{task:>10s}] {nbil.model_name:<24s} {format_metrics(m)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


def _metric_to_dict(m) -> dict:
    d = asdict(m)
    d["confusion"] = list(d["confusion"])
    return d


def _serialize(r) -> dict:
    return {
        "model_name": r.model_name,
        "task_kind": r.task_kind,
        "val": {k: _metric_to_dict(v) for k, v in r.val_metrics.items()},
        "test": {k: _metric_to_dict(v) for k, v in r.test_metrics.items()},
        "history": r.history,
        "checkpoint": str(r.checkpoint_path) if r.checkpoint_path else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
