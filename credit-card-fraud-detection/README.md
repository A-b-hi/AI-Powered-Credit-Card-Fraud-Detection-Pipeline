# 🛡️ AI Credit Card Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Random%20Forest-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

A comprehensive, end-to-end Machine Learning pipeline detecting fraudulent credit card transactions. This project includes extensive Exploratory Data Analysis (EDA), addresses severe class imbalance via SMOTE, compares 4 distinct ML algorithms, provides SHAP explainability, and wraps the best model into an interactive Web UI.

## 📁 Project Architecture

```
credit-card-fraud-detection/
├── data/                    
│   └── creditcard.csv       <-- Place Kaggle dataset here!
├── notebooks/               
│   ├── 01_eda.py            <-- Rich visualizations (Correlation, Class Imbalance)
│   ├── 02_train.py          <-- SMOTE preprocessing & Algorithm Comparison 
│   └── 03_explain.py        <-- SHAP visual explainability
├── models/                  
│   ├── best_model.pkl       <-- Auto-saved highest F1-score model
│   └── scaler.pkl           <-- Saved StandardScaler for live inference
├── app.py                   <-- Streamlit interactive live demo
└── requirements.txt         
```

## 🚀 Getting Started

### 1. Requirements

Ensure you have Python 3.8+ installed. Navigate to the project folder and run:

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
This code is written for the [European Credit Card Fraud dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). 

1. Download `creditcard.csv` from Kaggle.
2. Place it exactly at `credit-card-fraud-detection/data/creditcard.csv`.
*(Note: If you run the scripts without the dataset, they will auto-generate strict dummy data just to prove the pipeline compiles!)*

### 3. Run Pipeline Automatically
Run the scripts in order to train your model locally:

```bash
cd notebooks
python 01_eda.py       # Review dataset properties
python 02_train.py     # Train models on GPU/CPU and save the best one
python 03_explain.py   # Run SHAP on the trained model
cd ..
```

### 4. Launch the Live Web App
Once the model is trained and saved in `models/`, launch the interactive dashboard!

```bash
streamlit run app.py
```

## 📊 Methods Used
* **SMOTE (Synthetic Minority Over-sampling Technique)**: To mathematically balance the 0.17% fraud representation to 50/50.
* **XGBoost Classifier**: Ensembled gradient-boosted trees optimized for the local GTX 3050 GPU using `tree_method='hist', device='cuda'`.
* **SHAP (SHapley Additive exPlanations)**: Game theory approach to explain individual black-box ML predictions visually.
