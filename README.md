# Email Classification using TF-IDF with XGBoost and Naive Bayes

A side‑by‑side comparison of two classic machine learning pipelines for email spam detection: TF‑IDF + XGBoost and TF‑IDF + Naive Bayes.

## Overview

This project builds and compares two models for classifying emails as spam or ham (non‑spam):

- TF‑IDF + Naive Bayes
- TF‑IDF + XGBoost

The focus is on how a simple probabilistic model compares with a more expressive gradient‑boosted model when both use the same TF‑IDF text representation.

## Dataset

- Labeled email messages (spam vs ham).
- Standard text preprocessing: lowercasing, removing punctuation, stopword removal, and tokenization.
- Feature extraction with TF‑IDF on the cleaned text.
