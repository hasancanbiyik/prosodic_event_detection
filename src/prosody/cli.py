"""``prosody`` CLI entry point.

Examples
--------

Preprocess and cache the corpus to a single .npz::

    prosody preprocess --data-root AutoRPT_Data --out artifacts/corpus.npz

Run all four cells of the comparison matrix::

    prosody run-all --data-root AutoRPT_Data --out artifacts/results.json

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from prosody.data import load_corpus, speaker_split, summarize
from prosody.evaluate import format_metrics
from prosody.train import (
    train_frame_level_classical,
    train_frame_level_neural,
    train_sequence_level_classical,
    train_sequence_level_neural,
)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", required=True, help="Path to AutoRPT_Data/")
    p.add_argument("--max-files", type=int, default=None, help="Cap files (smoke runs)")
    p.add_argument("--val-speaker", default="m2b", help="Speaker held out for validation")
    p.add_argument("--test-speaker", default="f3a", help="Speaker held out for testing")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", default="artifacts")
    p.add_argument("-v", "--verbose", action="store_true")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def _load_split(args):
    corpus = load_corpus(args.data_root, max_files=args.max_files, progress=False)
    print(f"Loaded {len(corpus)} files", file=sys.stderr)
    stats = summarize(corpus)
    print(stats, file=sys.stderr)
    sp = speaker_split(corpus, val_speakers=[args.val_speaker], test_speakers=[args.test_speaker])
    print(
        f"Split sizes: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}",
        file=sys.stderr,
    )
    return sp, stats


def _serialize_classical(results) -> dict:
    out = {}
    for task, models in results.items():
        out[task] = {}
        for name, r in models.items():
            out[task][name] = {
                "val": _metric_to_dict(r.val),
                "test": _metric_to_dict(r.test),
            }
    return out


def _serialize_neural(r) -> dict:
    return {
        "model_name": r.model_name,
        "task_kind": r.task_kind,
        "val": {k: _metric_to_dict(v) for k, v in r.val_metrics.items()},
        "test": {k: _metric_to_dict(v) for k, v in r.test_metrics.items()},
        "history": r.history,
        "checkpoint": str(r.checkpoint_path) if r.checkpoint_path else None,
    }


def _metric_to_dict(m) -> dict:
    d = asdict(m)
    d["confusion"] = list(d["confusion"])
    return d


def cmd_run_all(args):
    sp, stats = _load_split(args)

    results: dict = {
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

    print("\n[1/4] Frame-level classical ML", file=sys.stderr)
    cf = train_frame_level_classical(sp.train, sp.val, sp.test, seed=args.seed)
    results["frame_classical"] = _serialize_classical(cf)
    for task, models in cf.items():
        for name, r in models.items():
            print(f"  {format_metrics(r.test)}  ← {name} [{task}]")

    print("\n[2/4] Sequence-level classical ML", file=sys.stderr)
    cs = train_sequence_level_classical(sp.train, sp.val, sp.test, seed=args.seed)
    results["sequence_classical"] = _serialize_classical(cs)
    for task, models in cs.items():
        for name, r in models.items():
            print(f"  {format_metrics(r.test)}  ← {name} [{task}]")

    print("\n[3/4] Frame-level neural (Conv1D s2s)", file=sys.stderr)
    nf = train_frame_level_neural(
        sp.train,
        sp.val,
        sp.test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    results["frame_neural"] = _serialize_neural(nf)
    for task, m in nf.test_metrics.items():
        print(f"  {format_metrics(m)}  ← {nf.model_name} [{task}]")

    print("\n[4/4] Sequence-level neural (CNN + BiLSTM)", file=sys.stderr)
    ncnn = train_sequence_level_neural(
        sp.train,
        sp.val,
        sp.test,
        kind="cnn",
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    nbil = train_sequence_level_neural(
        sp.train,
        sp.val,
        sp.test,
        kind="bilstm",
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    results["sequence_neural_cnn"] = _serialize_neural(ncnn)
    results["sequence_neural_bilstm"] = _serialize_neural(nbil)
    for r in (ncnn, nbil):
        for task, m in r.test_metrics.items():
            print(f"  {format_metrics(m)}  ← {r.model_name} [{task}]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote results to {args.out}", file=sys.stderr)


def cmd_preprocess(args):
    corpus = load_corpus(args.data_root, max_files=args.max_files, progress=True)
    stats = summarize(corpus)
    print(json.dumps(asdict(stats), indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            file_ids=np.array([c.file_id for c in corpus]),
            speakers=np.array([c.speaker for c in corpus]),
            **{f"feat_{c.file_id}": c.features for c in corpus},
            **{f"prom_{c.file_id}": c.prominence_labels for c in corpus},
            **{f"bound_{c.file_id}": c.boundary_labels for c in corpus},
        )
        print(f"Wrote {args.out}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prosody")
    sub = p.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("preprocess", help="Run preprocessing and dump stats / cache")
    pre.add_argument("--data-root", required=True)
    pre.add_argument("--max-files", type=int, default=None)
    pre.add_argument("--out", default=None)
    pre.add_argument("-v", "--verbose", action="store_true")
    pre.set_defaults(func=cmd_preprocess)

    runa = sub.add_parser("run-all", help="Run all four comparison cells")
    _add_common(runa)
    runa.add_argument("--out", default="artifacts/results.json")
    runa.set_defaults(func=cmd_run_all)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
