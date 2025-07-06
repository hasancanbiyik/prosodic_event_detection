# Prosodic Event Detection with Classical and Neural Models 
### [Research is in progress as of June 5, 2025]

## Overview
End-to-end pipeline for detecting prosodic prominence and boundaries in speech using traditional machine learning approaches.

Automatic detection of prosodic prominence and boundaries in speech using classical machine learning and neural networks. According to the findings, the neural networks (single layer CNN and RNN) achieve state-of-the-art performance on the Boston University Radio News Corpus through sequence-level classification and smart threshold optimization.

## Dataset
Boston University Radio News Corpus (AutoRPT)
- 142 audio files with hand-labeled prosodic events
- 16 acoustic features extracted per 10ms frame
- ~420,000 total frames

### Dataset Access

This project uses the [Boston University Radio News Corpus (AutoRPT)](https://catalog.ldc.upenn.edu/LDC96S36) for prosodic event detection.

To run the code, you must download and place the dataset. Due to licensing restrictions, I cannot host the dataset directly in this repository.

## Features
- Comprehensive acoustic feature extraction pipeline
- Class imbalance handling through sequence-level modeling
- Multi-task learning for joint prominence and boundary detection
- Reproducible experiments with detailed evaluation metrics

## Results
- **Prominence**: F1 = 0.47-0.48 (strong baseline)
- **Boundary**: F1 = 0.12-0.13 (more challenging task)
- Models generalize well across speakers
- Neural CNN: 85.3% prominence F1, 56.6% boundary F1
- 117% improvement over classical ML baselines
- Production-ready models for text-to-speech and speech recognition applications


| Model | Prominence F1 | Boundary F1 | Average F1 |
|-------|---------------|-------------|------------|
| Neural CNN | 85.3% | 56.6% | 71.0% |
| Neural RNN | 83.5% | 48.6% | 66.1% |
| Random Forest | 48.6% | 17.0% | 32.8% |
| Logistic Regression | 47.5% | 14.3% | 30.9% |

## Usage
1. Run `preprocess_data.ipynb` to extract features and create datasets
2. Run `traditional_ML_model_experiments.ipynb` to train and evaluate models
3. Run `neural_networks.ipynb` to train CNN/RNN models and evaluate final performance.


## Next Steps
- Experimenting with LLMs, fine-tuning models for improved performance

### Tags: #speech-processing, #prosody, #neural-networks, #machine-learning, #nlp, #pytorch, #speech-recognition, #text-to-speech
