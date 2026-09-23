# SMS Spam Detection

A deep learning model that classifies SMS messages as **spam** or **ham** (not spam), built with TensorFlow/Keras and trained on the classic SMS Spam Collection dataset.

## Overview

Given a raw SMS text message, the model predicts whether it's spam or legitimate. The pipeline covers the full workflow from raw text to a saved, reusable model:

1. **Load and clean** the dataset with `pandas`.
2. **Tokenize and pad** message text into fixed-length numeric sequences.
3. **Split** data into train/test sets.
4. **Build and train** a neural network for binary classification.
5. **Save** the trained model and tokenizer for reuse (via Keras's native format and `pickle`).

## Tech Stack

- **pandas** — load and clean the dataset
- **Tokenizer + pad_sequences** (`tensorflow.keras.preprocessing`) — convert raw text into padded numeric sequences
- **train_test_split** (`scikit-learn`) — split data into train/test sets
- **Sequential + Embedding + Dense** (`tensorflow.keras`) — build the spam detection model
- **pickle** — save the trained tokenizer for reuse
- **Keras native format (`.keras`)** — save the trained model for reuse

## Dataset

The [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 labeled SMS messages (`ham` or `spam`), loaded from `spam_SMS_dataset.csv`.

- Columns `v1` (label) and `v2` (message text) are kept; label is mapped to `0` (ham) / `1` (spam).
- Average message length: ~15.6 words.


## Results

Trained for 10 epochs on an 80/20 train/test split.

| Metric | Value |
|---|---|
| Test Accuracy | **98.39%** |
| Test Loss | 0.0521 |

## Project Structure

```
SMS-Spam-Detection/
├── SMS-Spam-Detection.ipynb   # Data prep, training, and evaluation
├── spam_SMS_dataset.csv       # Dataset (not included — see Setup)
├── model_spam_detector.keras  # Saved trained model
├── model_tokenizer.pkl        # Saved fitted tokenizer
├── app.py                     # Streamlit app for inference
└── requirements.txt
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and place it in the project root as `spam_SMS_dataset.csv`.

## Usage

### Train the model

Run through `SMS-Spam-Detection.ipynb` to clean the data, train the model, and save:
- `model_spam_detector.keras` — the trained model
- `model_tokenizer.pkl` — the fitted tokenizer (needed to preprocess new text the same way at inference time)

### Run the app

```bash
streamlit run app.py
```

## Notes

- The tokenizer must be saved and reused at inference time — fitting a new tokenizer on different text would produce a different word-to-index mapping than the one the model was trained on, breaking predictions.
- `input_length=50` at training time means messages are padded or truncated to 50 tokens; the same `maxlen=50` padding must be applied to any new text before prediction.
