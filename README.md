# IMDB Sentiment Classification with BERT and GPT-1

This repository contains an end-to-end NLP project for binary sentiment classification on the IMDB movie review dataset (`aclImdb`).  
Two transformer models are fine-tuned and compared:

- `bert-base-uncased` (BERT)
- `openai-gpt` (GPT-1)

The project includes data loading, tokenization, training loops, evaluation metrics, checkpointing, and final model export.

## Project Overview

The goal is to classify movie reviews as:

- `0` -> negative
- `1` -> positive

Both models are trained with similar hyperparameters to make comparison easier. During training:

- Validation metrics are computed each epoch.
- The best checkpoint is selected by **validation F1 score**.
- Final model artifacts are saved for later inference or deployment.

## Repository Structure

```text
.
|-- aclImdb/                         # IMDB dataset (train/test with pos/neg folders)
|-- tokenization/
|   |-- bert_tokenization.py         # BERT tokenizer + PyTorch Dataset
|   |-- gpt_tokenization.py          # GPT tokenizer + PyTorch Dataset
|-- training_models/
|   |-- bert_fine_tune.py            # BERT training pipeline
|   |-- gpt_fine_tune.py             # GPT-1 training pipeline
|-- data_loader.py                   # Reads dataset and creates train/val/test DataFrames
|-- evaluate_model.py                # Loss + accuracy/precision/recall/F1 evaluation
|-- save_checkpoint.py               # Checkpoint + metrics writer
|-- checkpoints_bert/                # Saved BERT checkpoints
|-- checkpoints_gpt1/                # Saved GPT-1 checkpoints
|-- final_bert_imdb/                 # Final BERT model/tokenizer files
|-- final_gpt1_imdb/                 # Final GPT-1 model/tokenizer files
|-- requirements.txt
|-- README.md
```

## How It Works

### 1) Data Loading

- `data_loader.py` reads `aclImdb/train/{pos,neg}` and `aclImdb/test/{pos,neg}`.
- Training split is further divided into train/validation (`val_size=0.2`) with stratification.

### 2) Tokenization and Dataset Objects

- `BERTIMDBDataset`: fixed-length tokenization (`max_length=256`) for BERT.
- `GPTIMDBDataset`: fixed-length tokenization for GPT-1.
- GPT-1 pipeline adds a prompt pattern: `Review: <text> Sentiment:`.
- If GPT tokenizer has no pad token, `[PAD]` is added.

### 3) Training

Both training scripts:

- Use `AdamW` optimizer.
- Use linear learning-rate schedule with warmup (`WARMUP_RATIO=0.1`).
- Use gradient clipping (`max_norm=1.0`).
- Use AMP (`torch.amp.autocast`) when CUDA is available.
- Save best checkpoint by validation F1.

### 4) Evaluation

`evaluate_model.py` computes:

- Loss
- Accuracy
- Precision
- Recall
- F1

## Installation

### Prerequisites

- Python 3.9+ recommended
- (Optional but recommended) NVIDIA GPU + CUDA for faster training

### Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset

This project expects the extracted `aclImdb` folder to be placed in the repository root.  
The dataset is not included in this repository and should be downloaded separately from the official source:

- [Large Movie Review Dataset (IMDB)](https://ai.stanford.edu/~amaas/data/sentiment/)

After extraction, the folder structure should look like this:

```text
aclImdb/
  train/
    pos/
    neg/
    unsup/
  test/
    pos/
    neg/
```

Official dataset notes (Large Movie Review Dataset v1.0):

- The core labeled set contains **50,000** reviews:
  - **25,000 train** and **25,000 test**
  - Balanced sentiment labels (`pos`/`neg`)
- There are **50,000 additional unlabeled** reviews for unsupervised learning (`train/unsup/`).
- In the labeled sets:
  - Negative review rating: `<= 4/10`
  - Positive review rating: `>= 7/10`
  - Neutral ratings are excluded from train/test labels.
- Train and test sets are built from **disjoint movie sets**.
- Review files follow the naming convention: `[id]_[rating].txt`.
  - Example: `test/pos/200_8.txt` means id `200`, rating `8/10`, positive label.
- URL files are included as `urls_[pos,neg,unsup].txt` (line number corresponds to review id).
- Additional resources distributed with the dataset:
  - `.feat` files (LIBSVM sparse feature format)
  - `imdb.vocab`
  - `imdbEr.txt`

For `.feat` format details, see [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/).

## Citation

Please cite the original paper if you use this dataset in academic work:

```bibtex
@inproceedings{maas-etal-2011-learning,
    title = "Learning Word Vectors for Sentiment Analysis",
    author = "Maas, Andrew L.  and
      Daly, Raymond E.  and
      Pham, Peter T.  and
      Huang, Dan  and
      Ng, Andrew Y.  and
      Potts, Christopher",
    editor = "Lin, Dekang  and
      Matsumoto, Yuji  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2011",
    address = "Portland, Oregon, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P11-1015/",
    pages = "142--150"
}
```

## Usage

Run training for each model from repository root:

```bash
python training_models/bert_fine_tune.py
python training_models/gpt_fine_tune.py
```

Artifacts produced:

- Per-epoch checkpoints in `checkpoints_bert/` and `checkpoints_gpt1/`
- Final export in `final_bert_imdb/` and `final_gpt1_imdb/`

## Current Results (Validation Checkpoints)

Based on saved checkpoint metrics:

### BERT (`checkpoints_bert/epoch_2/metrics.txt`)

- Accuracy: `0.9170`
- Precision: `0.9074`
- Recall: `0.9288`
- F1: `0.9180`
- Loss: `0.2450`

### GPT-1 (`checkpoints_gpt1/epoch_2/metrics.txt`)

- Accuracy: `0.9124`
- Precision: `0.9049`
- Recall: `0.9216`
- F1: `0.9132`
- Loss: `0.3251`

## Technical Notes and Evaluation

Project strengths:

- Clear modular structure (data, tokenization, training, evaluation, checkpointing).
- Consistent metric tracking and best-model selection by F1.
- Reproducible saved artifacts for both model families.

Potential improvements:

- Add random seeds across all libraries for stronger reproducibility.
- Add dedicated inference script (`predict.py`) for quick single-text testing.
- Add experiment config support (YAML/JSON/CLI args) instead of fixed constants.
- Add test-set metrics persistence (e.g., JSON/CSV logs).
- Add basic unit tests for data loading and dataset classes.

## Requirements

From `requirements.txt`:

- `torch`
- `transformers`
- `pandas`
- `scikit-learn`
- `numpy`
