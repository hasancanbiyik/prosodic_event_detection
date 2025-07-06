# 🎙️ Prosodic Event Detection with Classical and Neural Models  
### Research in Progress (as of July 6, 2025)

## Overview
This project implements an end-to-end machine learning pipeline for detecting **prosodic events**—specifically **prominence** and **boundary** markers—in speech.

I used both **traditional ML algorithms** and **simple neural networks (CNN & RNN)** to tackle the task. The neural models achieved state-of-the-art performance on the **Boston University Radio News Corpus (AutoRPT)**, showcasing how even compact architectures can model prosodic structure effectively through **sequence-level classification** and **smart threshold optimization**.

## Dataset

This project uses the **[Boston University Radio News Corpus (AutoRPT)](https://catalog.ldc.upenn.edu/LDC96S36)**:
- 142 audio files with hand-labeled prosodic events  
- 16 acoustic features extracted every 10ms  
- ~420,000 total frames

**Note**: Due to licensing restrictions, I cannot include the dataset in this repo. You'll need to obtain it from LDC and place it in the appropriate folder structure (`AutoRPT_Data/`) to run the code.

## Features
- Acoustic feature extraction pipeline built from scratch
- Sequence-level modeling to handle temporal structure
- Multi-task learning: detect **prominence** and **boundaries** jointly
- Class imbalance mitigation
- Fully reproducible experiments with evaluation metrics

## Results

- **Prominence**: F1 = 0.47–0.48 (classical ML baseline)  
- **Boundary**: F1 = 0.12–0.13 (classical ML baseline — much harder task)  
- Neural CNN achieved **85.3%** prominence F1, **56.6%** boundary F1  
- **117% improvement** over classical ML models  
- Models generalized well across speakers

| Model              | Prominence F1 | Boundary F1 | Average F1 |
|-------------------|---------------|-------------|------------|
| Neural CNN         | **85.3%**     | **56.6%**   | **71.0%**   |
| Neural RNN         | 83.5%         | 48.6%       | 66.1%       |
| Random Forest      | 48.6%         | 17.0%       | 32.8%       |
| Logistic Regression| 47.5%         | 14.3%       | 30.9%       |

## Usage

```bash
# Step 1: Clone the repo
git clone https://github.com/hasancanbiyik/prosodic_event_detection.git
cd prosodic_event_detection

# Step 2: Create a virtual environment
conda create -n prosody-nn python=3.10
conda activate prosody-nn
pip install -r requirements.txt
