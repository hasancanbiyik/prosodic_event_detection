# 🎙️ Prosodic Event Detection with Classical and Neural Models

> Research notebook project for prominence and boundary classification in speech.

## Overview

This repository explores prosodic-event classification using recordings and annotations derived from the Boston University Radio News Corpus (AutoRPT). It contains notebooks for:

1. extracting frame-level acoustic features and constructing binary labels;
2. evaluating classical frame-level classifiers; and
3. evaluating multi-task neural models on overlapping 500 ms windows.

The repository contains saved notebook outputs from the original experiments. The licensed source data, processed dataset, and trained checkpoints are not included, so the reported results cannot currently be independently reproduced from this repository alone.

## Dataset and Preprocessing

The preprocessing notebook's saved output reports:

- **142 audio files** and **284 annotation CSV files**;
- approximately **70 minutes** of audio;
- **420,045 frames** at a 10 ms frame shift; and
- **16 acoustic features** per frame.

The 16 features are extracted with Librosa and consist of:

- fundamental frequency (F0);
- RMS energy;
- spectral centroid; and
- 13 MFCCs.

Annotation timestamps from two files per recording are combined and expanded to a ±50 ms frame-labeling window. The resulting saved data contains 73,541 prominence-positive frames (17.51%) and 22,589 boundary-positive frames (5.38%).

The underlying Boston University Radio News Corpus is licensed and is not distributed in this repository. The preprocessing code expects WAV files and derived `*_annotations.csv` files containing `Prominence` and `Boundary` timestamp columns.

## Models

### Classical frame-level models

The classical notebook evaluates logistic regression, Gaussian Naive Bayes, decision trees, and random forests. Class weighting is used where supported, and features are standardized using statistics fitted on the training partition.

### Neural window-level models

The neural notebook constructs overlapping 50-frame windows and assigns prominence and boundary labels using fixed frame-proportion thresholds. It evaluates two multi-task architectures with shared representations and separate classification heads:

- a 1D convolutional neural network; and
- a bidirectional GRU with attention.

## Saved Notebook Results

These values are historical outputs embedded in the notebooks; they were not rerun during this repository review.

### Frame-level classical result

| Model | Evaluation | Prominence F1 | Boundary F1 |
|---|---|---:|---:|
| Shallow Random Forest | Held-out file partition | 0.471 | 0.123 |

The classical notebook also reports logistic-regression cross-validation scores of 0.480 ± 0.031 for prominence and 0.133 ± 0.011 for boundaries. That cross-validation is performed over frames rather than file or speaker groups and should therefore be treated as exploratory.

### 500 ms window-level neural results

| Model | Prominence F1 | Boundary F1 | Average F1 |
|---|---:|---:|---:|
| 1D CNN | **0.852** | **0.551** | **0.702** |
| Bidirectional GRU | 0.837 | 0.452 | 0.644 |

## Evaluation Caveats

- The classical and neural scores measure different targets: individual 10 ms frames versus thresholded 500 ms windows. They must not be used to calculate a direct percentage improvement.
- The train, validation, and test partitions are separated by file position but are not speaker-disjoint. The saved validation and test partitions both contain recordings from the `f3` speaker group.
- Neural training does not fix all random seeds, so exact results may vary between runs.
- The sequence-label thresholds are fixed design choices rather than the result of a documented optimization study.
- The notebooks do not implement a published AutoRPT event-scoring protocol or compare against external benchmarks. No state-of-the-art claim is made.
- A Random Forest model-selection cell reuses one estimator object across the two tasks. The later shallow Random Forest evaluation shown above retrains and evaluates each task separately and is the safer classical test result to cite.

## Repository Structure

```text
prosodic_event_detection/
├── media/                         # Screenshots from notebook runs
├── notebooks/
│   ├── preprocess_data.ipynb      # Feature extraction and frame labeling
│   ├── traditional_ML_models.ipynb
│   └── neural_networks.ipynb
├── LICENSE
├── README.md
└── requirements.txt
```

## Environment Setup

```bash
git clone https://github.com/hasancanbiyik/prosodic_event_detection.git
cd prosodic_event_detection

conda create -n prosody-nn python=3.10
conda activate prosody-nn

python -m pip install numpy pandas librosa seaborn soundfile \
  scikit-learn torch matplotlib jupyter
jupyter notebook
```

The existing `requirements.txt` is an informal dependency list rather than a locked, complete environment specification.

## Running the Notebooks

1. Arrange the licensed audio and derived annotation CSV files under an `AutoRPT_Data/` directory.
2. Update the hard-coded `data_root` path in `notebooks/preprocess_data.ipynb`.
3. Run `preprocess_data.ipynb` to create `autorpt_processed_subset.pkl`.
4. Keep the processed file in the working directory used by Jupyter.
5. Run `traditional_ML_models.ipynb` and `neural_networks.ipynb`.

## Project Status

This is an exploratory research project rather than a production event detector. A stronger evaluation would use speaker-disjoint splits, consistent frame- or event-level targets across every model, fixed random seeds, persisted split manifests, and a recognized prosodic-event scoring protocol.

## License

This repository's code is available under the [MIT License](LICENSE). The corpus data remains subject to its own licensing terms.
