# Sentiment Analysis: A Comparative Study of Traditional and Deep Learning Approaches

## Overview

This repository presents a comprehensive sentiment classification system designed to evaluate and compare the effectiveness of traditional machine learning (ML) and deep learning (DL) models.

We implemented:
- A **Logistic Regression** model using TF-IDF vectorization.
- Multiple **LSTM-based deep learning models** with various configurations and optimizers.

The goal was to identify which approach performs best for binary sentiment analysis on a large corpus of user-generated content.

---

## Dataset

We used the **IMDB Movie Reviews 50K** dataset from Kaggle, which contains:
- 50,000 labeled movie reviews
- Evenly split into training and testing sets
- Labels: `positive` or `negative`

[Dataset Source](https://www.kaggle.com/code/jillanisofttech/imdb-movie-reviews-50k/input)

---

## Preprocessing

- Text cleaning: lowercasing, punctuation removal
- Tokenization and stopword removal
- Padding for DL models (sequence length = 200)
- TF-IDF vectorization for ML model
- Embedded sequence representation for DL models (vocab size: 5000)

---

## Model Architectures

### Traditional ML
- **Logistic Regression**
- Input: 200-dimensional TF-IDF features

### Deep Learning (LSTM)
- Embedding Layer (trainable, 128 dims)
- One or two LSTM layers (with optional dropout)
- Dense layer with ReLU
- Output: Dense layer with sigmoid for binary classification

---

## Experimentation Summary

| Model ID | Optimizer | Dropout | Epochs | Accuracy | F1 Score | Notes |
|----------|-----------|---------|--------|----------|----------|-------|
| DL-003   | Nadam     | 0.4     | 20     | 0.8818   | 0.8809   | **Best performing model** |
| DL-001   | Adam      | 0.3     | 10     | 0.8720   | 0.8728   | Strong baseline |
| DL-004   | RMSprop   | 0.2     | 20     | 0.8688   | 0.8690   | Competitive |
| DL-005   | Nadam     | 0.2     | 20     | 0.8679   | 0.8678   | Initially selected |
| DL-002   | Adam      | 0.2     | 15     | 0.8618   | 0.8645   | Lower precision |
| Logistic Regression | - | - | - | 0.8600 | 0.8600 | Baseline model |

---

## Key Findings

- **DL-003** achieved the best results across all major performance metrics.
- The use of **dropout** and **Nadam optimizer** significantly contributed to its generalization.
- Logistic Regression remains efficient but was outperformed by all LSTM configurations in this task.

---

## Folder Structure

├── notebooks/

│ ├── Sentiment_analysis_G3_Adam.ipynb

│ ├── Sentiment_analysis_G3_Adam(dropout).ipynb

│ ├── Sentiment_analysis_G3_Nadam.ipynb

│ ├── Sentiment_analysis_G3_Nadam_dropout.ipynb

│ └── Sentiment_analysis_G3_RMSprop2.ipynb


├── models/

│ ├── model_nadam.keras

│ ├── model_adam.keras

│ ├── logistic_regression_model.pkl


├── report/

│ └── Sentiment Analysis Report - G3.pdf

---

## Team Contributions

- **Bernice** – Data preparation and preprocessing
- **Abubakar** – Logistic Regression model implementation
- **Purity** – LSTM model architecture design and tuning
- **Peter** – Report writing, structure, and final consolidation
