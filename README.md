# WHAT IF...? TRANSFORMERS WITH CAUSAL-AWARE EMBEDDINGS: SHAMELESSLY DEFORMING EMBEDDING SPACE

## Project Overview

This project explores the training and evaluation of causal embeddings for language models, specifically focusing on integrating these embeddings into GPT-2 architectures. The goal is to compare standard GPT-2 embeddings with embeddings trained on causal (cause-effect) data, and to analyze their impact on language modeling tasks.

## Features

- Training of causal embeddings using datasets of (cause, effect) pairs.
- Integration of causal embeddings into GPT-2 models.
- Comparison of training dynamics and performance between regular and causal-embedding GPT-2 models.
- Visualization and logging of results using Weights & Biases (wandb).
- Jupyter notebooks for data processing, embedding training, and model evaluation.

## Project Structure

```
implementation/
  datasets/                # Datasets and processing scripts
  evaluation/              # Evaluation outputs and scripts
  models/                  # Saved model weights (e.g., causal_embeds.pth)
  notebooks/               # Jupyter notebooks for training and analysis
  scripts/                 # Python scripts for model comparison
paper/                     # Plots and figures for publication
tex/                       # LaTeX source for paper
requirements.txt           # Python dependencies
README.md                  # Project documentation
```

## Installation

1. Clone the repository.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Datasets

- **e-CARE**: Used for training causal embeddings (see `causal_embeddings.ipynb`).
- **Wikitext-2**: Used for language modeling and comparison experiments.
- Additional datasets for causal reasoning are included in `implementation/datasets/`.

## Usage

### 1. Train Causal Embeddings

Open and run the notebook:
```
implementation/notebooks/causal_embeddings.ipynb
```
This notebook loads the e-CARE dataset and trains causal embeddings using PyTorch and HuggingFace Transformers.

### 2. Integrate and Compare in GPT-2

- Use the notebook:
  ```
  implementation/notebooks/attention_integration.ipynb
  ```
  to integrate causal embeddings into GPT-2 and compare with regular embeddings.

- Or run the script:
  ```
  python implementation/scripts/compare_gpt2_causal.py
  ```
  This script:
  - Loads Wikitext-2.
  - Sets up two GPT-2 models: one with standard embeddings, one with your trained causal embeddings.
  - Trains both models and evaluates them.
  - Saves training curves and metrics in `implementation/evaluation/`.

### 3. Visualization and Analysis

- Training and evaluation metrics are logged to wandb.
- Plots are saved in `paper/plots/` and `implementation/evaluation/`.

## Requirements

- Python 3.8+
- torch >= 2.0
- transformers >= 4.35
- datasets
- scikit-learn >= 1.3
- umap-learn >= 0.5
- matplotlib >= 3.7
- Pillow >= 10.0
- jupyter, ipykernel, pandas, wandb

See `requirements.txt` for the full list.

## Results

- Training loss and perplexity comparisons between regular and causal-embedding GPT-2 models.
- Visualizations of embedding spaces and model performance.
- All results and plots are available in the `paper/plots/` and `implementation/evaluation/` directories.
