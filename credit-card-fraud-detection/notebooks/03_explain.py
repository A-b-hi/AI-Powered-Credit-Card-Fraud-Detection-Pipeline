# Phase 3: Model Explainability (SHAP)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
from sklearn.model_selection import train_test_split

plt.style.use('dark_background')


# 1. Loading Data and Models

DATA_PATH = '../data/creditcard.csv'

if not os.path.exists(DATA_PATH) or not os.path.exists('../models/best_model.pkl'):
    print("⚠️ Dataset or Trained Model not found! To run SHAP:")
    print("1. Please make sure creditcard.csv is in the 'data' folder")
    print("2. Please run '02_train.py' first to generate the model and scaler.")
else:
    df = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully!")
    
    model = joblib.load('../models/best_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    print("✅ XGBoost Model and Scaler loaded successfully!")


    df['Amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.transform(df['Time'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42, stratify=y)
    

    print("Sampling 500 rows to compute SHAP explanations...")
    X_background = shap.sample(X_test, 500)
    
    # Initializing the Tree explainer 
    print("\n Calculating SHAP values. This may take a minute...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)

    print("✅ SHAP Values Computed!")

    # 2. SHAP Summary Plot
    print("\n[Plot 1] SHAP Summary Plot:")
    print("Look at this plot to understand GLOBAL feature importance.")
    plt.figure()
    shap.summary_plot(shap_values, X_background)
    

     # 3. Explaining Individual Prediction
    fraud_indices = np.where(y_test.loc[X_background.index] == 1)[0]
    
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        actual_data = X_background.iloc[fraud_idx]
        
        print(f"\n[Plot 2] Examining Fraudulent Transaction Number {fraud_idx}...")
        print(f"Model prediction score: {model.predict_proba(actual_data.values.reshape(1, -1))[0][1]:.4f}")
        
        
        plt.figure()
        shap.force_plot(
            explainer.expected_value, 
            shap_values[fraud_idx,:], 
            actual_data, 
            matplotlib=True,
            show=True
        )
    else:
        print("No fraud cases found in this random sample to plot. Run again or increase sample size!")
