# Credit Card Fraud Detection Using Deep Learning

This project applies a deep learning approach to detect fraudulent credit card transactions using real-world anonymized data. It uses a Multi-Layer Perceptron (MLP) built with TensorFlow and Keras to classify transactions as fraudulent or legitimate.

## 📊 Overview

- Fraudulent transactions are rare but costly.
- The dataset is highly imbalanced (less than 1% fraud cases).
- This project focuses on high **recall** and **precision** rather than just accuracy.

## 📁 Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Observations**: 284,807  
- **Features**: 30 (V1–V28 anonymized features, plus `Time`, `Amount`)  
- **Target**: `Class` (0 = Legitimate, 1 = Fraud)

## 🔧 Technologies Used

- **Python 3**
- **NumPy, Pandas** – Data manipulation
- **Matplotlib, Seaborn** – Visualizations
- **Scikit-learn** – Evaluation metrics
- **TensorFlow, Keras** – Neural network modeling
- **SMOTE (Imbalanced-learn)** – Class balancing

## 🔄 Workflow

1. **Data Exploration & Preprocessing**
   - Visualized class imbalance and feature correlations
   - Normalized `Amount` feature
   - Applied SMOTE to balance dataset

2. **Model Building**
   - MLP with two hidden layers (16 and 8 units) and a sigmoid output layer
   - Activation: ReLU (hidden layers), Sigmoid (output)
   - Loss: Binary crossentropy | Optimizer: Adam

3. **Model Evaluation**
   - Confusion matrix
   - ROC-AUC curve
   - Precision, Recall, F1 Score

## 📈 Results

| Metric        | Value (Approx.) |
|---------------|-----------------|
| Accuracy      | ~99%            |
| Recall (Fraud)| High            |
| AUC Score     | ~0.99           |

Emphasis was placed on **recall** to reduce the risk of false negatives — crucial in fraud detection systems.

## 📌 Future Improvements

- Try XGBoost or Random Forest for comparison  
- Implement dropout layers or L2 regularization  
- Deploy with Streamlit or Flask for real-time demo  

## 🧻 License

This project is for educational purposes only.  
Dataset provided by Worldline and ULB via [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## 👤 Author

**Arnold Muzarurwi**  
Master’s Student, Data Analytics & Visualization  
📫 [amuzarur@mail.yu.edu](mailto:amuzarur@mail.yu.edu)  
🌐 [LinkedIn](https://www.linkedin.com/in/arnold-muzarurwi-4681852b2) 




