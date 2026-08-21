# 💳 Fraud-Detection-in-Credit-Card-Transactions

<div align="center">

### Machine Learning | Classification | Streamlit

Detect fraudulent credit card transactions using machine learning techniques.

<br>

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit)

</div>

---

## 📊 Dataset Overview

| Feature | Value |
|---------|-------|
| Total Transactions | 284,807 |
| Total Features | 30 |
| Fraud Cases | 473 (0.17%) |
| Legitimate Cases | 284,334 (99.83%) |
| Data Status | Highly Imbalanced ⚠️ |

---

## 🎯 Project Approach

1. **Data Cleaning**
   - Removed 1,081 duplicate records
   - Verified data integrity

2. **Exploratory Data Analysis**
   - Univariate, bivariate analysis
   - Distribution and skewness analysis
   - Correlation analysis

3. **Feature Engineering**
   - Yeo-Johnson transformation for `Amount` feature
   - Standard scaling for `Time` and `Amount`

4. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - RandomUnderSampling
   - Class weights balancing

5. **Model Training & Comparison**
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier

6. **Model Selection**
   - Random Forest with SMOTE selected as final model

7. **Application**
   - Streamlit web application
   - Model serialization with Joblib

---

## 🏆 Final Model Performance

**Selected Model:** Random Forest + SMOTE

| Metric | Score |
|--------|-------|
| 🎯 Precision | 0.77 |
| 📈 Recall | 0.79 |
| ⭐ F1-Score | 0.78 |
| 📊 ROC-AUC | 0.98 |

---
---

## ℹ️ Note on V1–V28

`V1–V28` are anonymized PCA-transformed features provided by the original dataset. The Streamlit application accepts the same feature structure used during model training rather than asking users to manually enter these anonymized features.

## 🖥️ Application Screenshots
![alt text](screenshots/image.png)
![alt text](screenshots/image-1.png)
![alt text](screenshots/image-2.png)

## 🚀 How to Run

### Prerequisites
- Python 3.7+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Streamlit app:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| 🐍 Python | Core language |
| 🐼 Pandas | Data manipulation |
| 🔢 NumPy | Numerical operations |
| 🤖 Scikit-learn | Machine learning |
| ⚖️ Imbalanced-learn | SMOTE implementation |
| 📊 Matplotlib | Data visualization |
| 🎨 Seaborn | Statistical visualization |
| 💾 Joblib/Pickle | Model serialization |
| 🌐 Streamlit | Web application |

---

## 📈 Model Comparison

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.67 | 0.87 | 0.76 | 0.85 |
| Decision Tree | 0.62 | 0.82 | 0.71 | 0.80 |
| **Random Forest** | **0.77** | **0.79** | **0.78** | **0.98** |

---

## 🔑 Key Features

- ✅ Handles highly imbalanced dataset
- ✅ Multiple resampling techniques
- ✅ High ROC-AUC score (0.98)
- ✅ Interactive web interface
- ✅ Real-time predictions
- ✅ Probability estimates

---
