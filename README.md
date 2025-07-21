# 🧠 Dysarthria Severity Classification

**Classify dysarthric speech severity using Wav2Vec2 and DistilAlHuBERT embeddings.**

---

## 📌 Overview

Dysarthria is a motor speech disorder that affects articulation and intelligibility. This project aims to **automatically classify the severity level** of dysarthric speech (e.g., *mild*, *moderate*, *severe*) using pre-trained **self-supervised speech models** — **Wav2Vec2** and **DistilAlHuBERT** — for feature extraction, followed by deep learning or traditional models for classification.

---

## 📂 Datasets

### ✅ Tamil Dysarthric Speech Corpus (SSN-TDSC)

* **Language**: Tamil
* **Content**: 365 phonetically balanced sentences and isolated-words
* **Speakers**: Mild and moderate dysarthric speakers
* **Format**: `.wav` audio files with corresponding labels

### ✅ UA-Speech

* **Language**: English
* **Content**: Isolated word corpus from dysarthric and control speakers
* **Speakers**: 15 dysarthric and 13 control speakers

### ✅ TORGO Corpus

* **Language**: English
* **Content**: Sentences, isolated words, and paragraph-level recordings
* **Speakers**: 8 Dysarthric and 7 control speakers with metadata for severity

> 🔐 *These datasets were collected under ethical guidelines. Ensure proper attribution and usage rights.*

---

## 🧠 Methodology

### 1. **Feature Extraction**

* **Wav2Vec2**: Extracted using `facebook/wav2vec2-base`
* **DistilAlHuBERT**: Extracted using `voidful/distilalhubert`
* Output: `.npy` files and `.pt`  files containing embeddings per utterance

### 2. **Preprocessing**

* Mean pooling across frames to get a fixed-size embedding
* Normalization

### 3. **Classification**

* Models used:

  * Support Vector Machines (SVM)
  * Convolutional Neural Networks (CNN)
* Hyperparameter tuning (e.g., `C`, `gamma` for SVM; learning rate, dropout for CNN)
* Evaluation:

  * Accuracy
  * F1-score
  * Confusion Matrix

### 4. **Cross-Validation Strategy**

* **Leave-One-Speaker-Out (LOSO)** cross-validation used to evaluate generalizability
* For each iteration:

  * One speaker is held out for testing
  * Remaining speakers used for training
  * Metrics averaged across all iterations

---

## 🧠 Cross-Dataset Strategy

* Independent training on datasets: SSN-TDSC, UA-Speech, TORGO
* Balanced data preprocessing across corpora
* Domain adaptation explored for generalization across languages

---

## 📊 Results

| Model | Feature Set    | Accuracy | F1-Score |
| ----- | -------------- | -------- | -------- |
| SVM   | Wav2Vec2       |   |      |
| CNN   | Wav2Vec2       |   |      |
| SVM   | DistilAlHuBERT |    |     |
| CNN   | DistilAlHuBERT |    |     |

> 📈 Best performance achieved using **Wav2Vec2 + CNN** with **LOSO** cross-validation

---

## 📜 Folder Structure

```
dysarthria_severity_classification/
│
├── audio/                  # Raw audio files
├── embeddings/             # Extracted embeddings (.npy)
├── labels.csv              # Severity labels
├── extract_embeddings.py   # Embedding extraction script
├── train_classifier.py     # Model training and evaluation
├── utils.py                # Helper functions
└── README.md
```

---

## 🤝 Acknowledgements

* **SSN-TDSC**, **UA-Speech**, and **TORGO** corpora
* **Facebook AI** for Wav2Vec2
* **Voidful** for DistilAlHuBERT
* **NIEPMD** for clinical guidance
