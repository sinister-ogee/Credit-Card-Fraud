# Credit Card Fraud Detection Using Deep Learning

This project implements a deep learning approach to detect fraudulent credit card transactions using real-world anonymized data. A Multi-Layer Perceptron (MLP) built with TensorFlow and Keras was trained, tested, and compared against a Random Forest baseline to evaluate effectiveness on imbalanced data.

---

## 📊 Overview
- Fraudulent transactions represent less than 0.2% of the dataset, making class imbalance a major challenge.  
- Two MLP architectures were tested: one with a **single hidden layer** and another with **two hidden layers**.  
- Performance was evaluated against Random Forest to test the hypothesis that deeper MLPs yield significant improvements in fraud detection.  

---

## 📁 Dataset
- **Source**: Kaggle – Credit Card Fraud Detection  
- **Observations**: 284,807 transactions  
- **Features**: 30 (V1–V28 anonymized PCA features, plus standardized Amount, excluding Time)  
- **Target**: Class (0 = Legitimate, 1 = Fraud)  

---

## 🔧 Technologies Used
- **Programming**: Python 3  
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
- **Deep Learning**: TensorFlow, Keras  
- **Imbalanced Data Handling**: ADASYN oversampling, random undersampling  
- **Evaluation**: Accuracy, F1-score, ROC-AUC, Wilcoxon Signed-Rank Test  

---

## 🔄 Workflow
### Data Preprocessing
- Dropped non-informative *Time* feature  
- Standardized *Amount* feature using RobustScaler  
- Split into training, validation, and test sets with stratified sampling  
- Balanced training data using **ADASYN oversampling** and controlled undersampling  

### Model Building
- Implemented MLP models with 1 and 2 hidden layers  
- Tuned hyperparameters (neurons, dropout, initialization, learning rate) via grid search  
- Optimizer: Adam | Loss: Binary Crossentropy | Activation: ReLU (hidden), Sigmoid (output)  

### Model Evaluation
- Compared **single-layer vs dual-layer MLPs** over 60+ runs  
- Benchmarked against **Random Forest** results  
- Applied **Wilcoxon Signed-Rank Test** to assess statistical significance  

---

## 📈 Results
- **F1-score (dual-layer MLP):** ~0.77 (mean across multiple runs)  
- Statistically significant performance difference between single- and dual-layer MLPs (p < 0.05)  
- Models with dropout showed improved generalization on validation data  

---

## 📌 Future Improvements
- Explore advanced ensemble methods (XGBoost, Random Forest tuning)  
- Deploy trained model as an API using Flask or FastAPI  
- Integrate with a real-time fraud monitoring pipeline  

---

## 👤 Author
**Arnold Muzarurwi**  
Master’s Student, Data Analytics & Visualization  
📫 amuzarur@mail.yu.edu | 🌐 LinkedIn  


