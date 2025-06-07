# Credit Card Fraud Detection Using Neural Networks

This project applies a deep learning approach to detect fraudulent credit card transactions using real-world anonymized data. It uses a Multi-Layer Perceptron (MLP) model built with TensorFlow and Keras to classify transactions as fraudulent or legitimate.

## 📊 Overview

- Fraudulent transactions are rare but costly.
- The dataset is highly imbalanced (less than 1% fraud cases).
- This project focuses on high **recall** and **precision** rather than just accuracy.

## 📁 Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Observations**: 284,807  
- **Features**: 30 (V1–V28 anonymized features, plus `Time`, `Amount`)  
- **Target**: `Class` (0 = Legit, 1 = Fraud)

## 🔧 Technologies Used

- **Python 3**
- **NumPy, Pandas** – Data manipulation
- **Matplotlib, Seaborn** – Visualizations
- **Scikit-learn** – Evaluation metrics
- **TensorFlow, Keras** – Neural network
- **SMOTE (from imbalanced-learn)** – Class balancing

## 🔄 Workflow

1. **Data Exploration & Preprocessing**
   - Feature scaling on `Amount`
   - SMOTE used to balance class distribution

2. **Model Building**
   - 3-layer MLP with ReLU activations
   - Sigmoid output layer for binary classification
   - Binary cross-entropy loss + Adam optimizer

3. **Model Evaluation**
   - Confusion matrix
   - ROC-AUC curve
   - Precision, Recall, F1 Score

## 📈 Results

| Metric        | Value |
|---------------|--------|
| Accuracy      | ~99%  |
| Recall (Fraud)| High  |
| AUC Score     | ~0.99 |

Emphasis was placed on **recall** to reduce the risk of false negatives — crucial in fraud detection.



