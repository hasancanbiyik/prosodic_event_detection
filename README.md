# Prosodic Event Detection on the Boston University Radio News Corpus

Detect prosodic **prominence** (pitch accents) and **boundary** (intonational
phrase tones) on the AutoRPT slice of the Boston University Radio News Corpus
(BURNC). Compares classical machine-learning baselines against compact
neural models on **two tasks**: per-frame (10 ms) and per-window (500 ms).

> **Status, August 2026.** This is the v0.2 rewrite. The original (v0.1)
> notebooks shipped a misleading classical-vs-neural headline because the
> two model families were evaluated on different tasks. The numbers below are
> from a fair comparison and a speaker-aware split. The original notebooks are
> archived under [`notebooks/_archive_original/`](notebooks/_archive_original/)
> for transparency.

---

## What "prosodic event detection" means

Prosody is the music of speech — pitch, timing, loudness — that listeners use
to figure out which words are emphasised and where one phrase ends and the
next begins. Computational prosody is the engineering version of that
intuition. We turn an audio waveform into a 16-dimensional feature stream
(F0, energy, spectral centroid, MFCCs) every 10 ms and ask:

* **Prominence (pitch accent)** — is *this* moment in the audio one that a
  human ToBI annotator marked with `H*`, `L+H*`, `!H*`, etc.?
* **Boundary (intonational phrase boundary)** — is *this* moment one of those
  drop-the-pitch-and-take-a-breath spots, marked with labels ending in `%`
  (e.g. `L-L%`, `L-H%`, `H-L%`)?

Think of frame-level detection like a metal detector sweeping over a beach
every 10 ms; sequence-level detection bunches the audio into 500 ms tiles and
asks "did the metal detector beep anywhere inside this tile?" Same underlying
signal, very different baselines.

## Dataset

[Boston University Radio News Corpus / AutoRPT](https://catalog.ldc.upenn.edu/LDC96S36).
Six speakers (`f1a, f2b, f3a, m1b, m2b, m3b`), 142 audio files,
~70 minutes, ~420 K frames at 10 ms. Labels come from the per-file `.ton`
files (ToBI annotation), parsed by [`prosody.labels.ToBIParser`][parser].

| Frame positive rate (all files) | Sequence positive rate (held-out test speaker) |
| --- | --- |
| Prominence: 17.5 % | 66.6 % |
| Boundary: 5.3 %    | 22.9 % |

The dataset itself is not redistributed here — obtain it from LDC and place
it in `AutoRPT_Data/` with the per-speaker folder layout
(`AutoRPT_Data/<speaker>/<style>/<file>.{wav,ton,TextGrid}`).

## Why the v0.1 results were misleading

The original notebooks had several methodological and implementation problems:

1. **Apples-to-oranges comparison.** Classical ML was trained per frame
   (positive rate ≈ 17 % prominence / 5 % boundary). The CNN/RNN were trained
   on 50-frame (500 ms) windows where a window was labelled positive iff at
   least 15 %/10 % of its frames were positive. That re-labeling pushed the
   positive rate to 67 %/23 %, making the task much easier. A
   trivial always-positive classifier on the sequence-level prominence task
   already scores F1 ≈ 0.80. The original headline "85.3 % CNN F1" was only
   ~5 points above that trivial baseline.
2. **No speaker-aware split.** Downstream notebooks sliced the processed list
   by file order, so speakers were not guaranteed to be disjoint across train,
   validation, and test.
3. **Preprocessing leakage.** The neural notebook fitted its feature scaler on
   the full corpus, including validation and test features. The rewrite fits
   normalization on training data only.
4. **Estimator/checkpoint reuse.** The original classical test cell could use
   a boundary-fitted Random Forest for prominence, which explains the reported
   0.044 prominence F1; a separately refitted simple RF reached 0.471 in the
   same notebook. The CNN and GRU also wrote to the same checkpoint filename.

The rewrite reports **both** tasks for **both** model families and uses a
**speaker-aware** train/val/test split (default: 4 speakers train,
`m2b` val, `f3a` test).

## Headline numbers

Numbers below are from `scripts/run_classical_only.py` and
`scripts/run_neural.py` on the speaker-aware split (val=m2b, test=f3a).
Random seed 42. Full JSON outputs live under `artifacts/`.

### Frame-level (per 10 ms frame, the harder task)

| Model | Prominence F1 | Boundary F1 | Notes |
| --- | --- | --- | --- |
| Trivial (always positive)        | 0.296 | 0.093 | from positive rate alone |
| LogisticRegression (frame-level) | 0.485 | 0.118 | this rewrite |
| RandomForest (frame-level)       | 0.472 | 0.135 | this rewrite |
| FrameClassifier Conv1D (s2s)     | **0.562** | **0.251** | this rewrite |

The frame-level Conv1D s2s model — which the v0.1 notebooks did not have —
clears the strongest classical baseline by approximately **+8 F1 points** on
prominence and **+12 points** on boundary. AUC is 0.877 / 0.821 respectively;
the relatively low precision (0.429 / 0.157) is consistent with the use of
class-weighted loss and a fixed 0.5 decision threshold. Each of the 71,802 test
frames is evaluated exactly once even though overlapping windows are used for
training.

### Sequence-level (per 500 ms window, the original CNN task)

| Model | Prominence F1 | Boundary F1 | Notes |
| --- | --- | --- | --- |
| Trivial (always positive)             | 0.800 | 0.373 | from positive rate alone |
| LogisticRegression (pooled features)  | 0.808 | 0.467 | this rewrite |
| RandomForest (pooled features)        | 0.806 | 0.407 | this rewrite |
| Sequence CNN (this rewrite)           | **0.842** | **0.561** | speaker-aware split |
| Sequence BiLSTM (this rewrite)        | 0.840 | 0.560 | speaker-aware split |
| Original CNN (v0.1, *file-order* split) | 0.852 | 0.551 | reference only — different split/leakage |
| Original BiGRU (v0.1, *file-order* split) | 0.837 | 0.452 | reference only — different split/leakage |

The honest picture: on sequence-level **prominence**, classical pooled-feature
models sit right at the trivial baseline (0.808 vs 0.800). The neural models
clear it by ~+4 points — real but small. The v0.1 "117 % improvement"
headline was framed against frame-level classical numbers (0.47) on a
strictly harder task; the proper apples-to-apples lift is closer to
**+4 % relative**, not 117 %.

On sequence-level **boundary**, the CNN beats LR by approximately **+9 points**
(0.561 vs 0.467), which is the clearest neural-vs-classical signal in the
sequence task.

## Project layout

```
.
├── AutoRPT_Data/                   # ← LDC corpus (not redistributed)
├── src/prosody/                    # the package
│   ├── data.py                     # corpus loading, speaker-aware splits
│   ├── features.py                 # 16-dim librosa features
│   ├── labels.py                   # ToBI parser, frame/sequence labels
│   ├── models.py                   # FrameClassifier / ProminenceCNN / BiLSTM
│   ├── _train_classical.py         # classical ML (no torch dep)
│   ├── train.py                    # neural training (torch)
│   ├── evaluate.py                 # metrics + trivial baselines
│   └── cli.py                      # `prosody` CLI entry point
├── notebooks/
│   ├── 01_preprocessing_demo.ipynb
│   ├── 02_classical_demo.ipynb
│   ├── 03_neural_demo.ipynb
│   └── _archive_original/          # the pre-rewrite notebooks
├── scripts/
│   ├── run_classical_only.py       # the no-torch experiment runner
│   └── run_neural.py               # the neural experiment runner
├── tests/                          # pytest suite (labels, features, models)
├── artifacts/                      # tracked JSON results; local caches/checkpoints ignored
├── pyproject.toml
└── Makefile
```

## Quickstart

```bash
git clone https://github.com/hasancanbiyik/prosodic_event_detection.git
cd prosodic_event_detection

# uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,notebooks]"

# or conda + pip
conda create -n prosody python=3.10 -y
conda activate prosody
pip install -e ".[dev,notebooks]"

# Place AutoRPT_Data/ at the repo root, then:
make classical          # ~30 s preprocessing + ~30 s classical ML
make neural             # ~10–15 minutes on CPU; faster on MPS / CUDA
make all                # classical + neural

pytest -q               # 36 tests; neural smoke tests require torch
```

The `prosody` console script also works once installed:

```bash
prosody preprocess --data-root AutoRPT_Data --out artifacts/corpus.npz
prosody run-all --data-root AutoRPT_Data --out artifacts/all_results.json -v
```

## Caveats and known limitations

* **Single held-out test speaker.** Six speakers, three female / three male.
  We test on `f3a` and validate on `m2b`; numbers depend on which speaker you
  hold out. Leave-one-speaker-out cross-validation is the clean next step.
* **Threshold = 0.5 throughout.** With class-imbalanced BCE + `pos_weight`,
  the model output distribution does not centre at 0.5. The frame-level
  Conv1D's precision (0.429 prominence, 0.157 boundary) at AUC 0.877 / 0.821
  indicates useful ranking but a poorly calibrated operating point. Tuning the
  classification threshold on validation data may improve frame-level F1.
* **Neural training is unstable on this small corpus.** The reported JSON is
  one seed-42 run, with checkpoints selected by mean validation F1 across both
  tasks. Sequence validation scores fluctuate substantially across epochs and
  may vary somewhat with PyTorch version and accelerator backend.
* **Overlapping training windows.** Frame-level neural training uses 50-frame
  windows with 50% overlap, so interior training frames receive more weight
  than edge frames. Validation and test inference operate on complete files,
  ensuring each reported frame is counted once.
* **`librosa.yin` with `fmin=50` Hz and `frame_length=400` samples** prints a
  warning that less than two pitch periods fit in a frame. We keep the
  original parameters for direct comparability with v0.1; raising `fmin` to
  ~80 Hz would silence the warning at the cost of clipping low-pitched
  speakers.
* **Sequence-level positive thresholds** (15 % prominence, 10 % boundary) are
  inherited from the original notebook. They are exposed as keyword arguments
  on `build_sequence_level` and `train_sequence_level_*` so you can re-tune.
* **No external published baseline cited yet.** Adding a Rosenberg-style
  baseline number from the prosodic-event-detection literature would give
  readers an absolute reference point.

## Roadmap (where this is going)

The foundation rebuild is now done. Next up, in rough order of payoff for an
ML-engineering portfolio:

1. **CI** — GitHub Actions running `pytest`, `ruff`, `black --check`, plus a
   docker build job. Adds the green badge.
2. **FastAPI + Docker inference service** — POST a wav, get JSON
   frame-level predictions back.
3. **Streamlit / Gradio frontend** — upload a `.wav`, play it, and inspect
   waveform-aligned prominence/boundary overlays.
4. **Pre-commit hooks** (`black`, `ruff`, `mypy`, `nbstripout`).
5. **Experiment tracking** with Weights & Biases (public-project free tier).
6. **Model card** under `MODEL_CARD.md` documenting intended use and
   per-speaker performance.
7. **Self-supervised baseline** — fine-tune a frozen `wav2vec2-base` on
   prosodic event detection and compare against the hand-crafted feature
   models. This is the single highest-leverage technical addition.

## License

[MIT](LICENSE).

[parser]: src/prosody/labels.py
