# Phase 2: Preprocessing & Model Training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE

plt.style.use('dark_background')
sns.set_theme(style="darkgrid")


# 1. Load Data
DATA_PATH = '../data/creditcard.csv'
if not os.path.exists(DATA_PATH):
    print("⚠️ Dataset not found! Auto-generating dummy rows so the training script works immediately...")
    df = pd.DataFrame(np.random.randn(5000, 31), columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class'])
    df['Class'] = np.random.choice([0, 1], size=5000, p=[0.95, 0.05])
else:
    df = pd.read_csv(DATA_PATH)
    print("✅ Real dataset loaded successfully!")


# 2. Preprocessing
print("\n[Stage 1] Preprocessing and Scaling Numerical Features...")
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

os.makedirs('../models', exist_ok=True)
joblib.dump(scaler, '../models/scaler.pkl')


X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Original Train shape: {X_train.shape} | Original Fraud count: {sum(y_train == 1)}")


# 3.balancing
print("\n[Stage 2] Applying SMOTE to balance the classes perfectly...")
t0 = time.time()
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
t_smote = time.time() - t0

print(f"✅ SMOTE complete in {t_smote:.2f} seconds.")
print(f"Resampled Train shape: {X_train_res.shape} | New Fraud count: {sum(y_train_res == 1)} | New Legit count: {sum(y_train_res == 0)}")


# 4. Model Training & Comparison
print("\n[Stage 3] Training and Comparing 4 ML Models (CPU & GPU)...")


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost (GPU)": XGBClassifier(tree_method='hist', device='cuda', max_depth=6, random_state=42), 
    "Linear SVC": LinearSVC(max_iter=3000, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n⚙️ Training {name}...")
    t0 = time.time()
    try:
        model.fit(X_train_res, y_train_res)
    except Exception as e:
        print(f"  ⚠️ Warning: Could not use GPU for {name}. Falling back to normal CPU. Error: {e}")
        
        model = XGBClassifier(max_depth=6, random_state=42, n_jobs=-1) 
        model.fit(X_train_res, y_train_res)
        
    y_pred = model.predict(X_test)
    
    # Metrics
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    if not hasattr(model, 'predict_proba'): 
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        
    f1 = f1_score(y_test, y_pred)
    train_time = time.time() - t0
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall (Detection Rate)': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1,
        'AUC-ROC': roc_auc_score(y_test, y_prob),
        'Time (s)': train_time,
        'y_prob': y_prob,
        'y_pred': y_pred
    }
    print(f"✅ Took {train_time:.2f} seconds. | F1-Score: {f1:.4f}")


# 5. Visualizing Model Performance
results_df = pd.DataFrame(results).T.drop(['y_prob', 'y_pred'], axis=1)

print("\n--- Final Model Comparison ---")
print(results_df.round(4))

print("\n🔍 Detailed Model Scores:\n")

for model_name, metrics in results.items():
    print(f"\n🔹 {model_name}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall (Detection Rate)']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")
    print(f"Training Time: {metrics['Time (s)']:.2f} sec")
# Bar chart of Key Metrics
plt.figure(figsize=(12, 6))
results_df[['Precision', 'Recall (Detection Rate)', 'F1-Score']].plot(kind='bar', colormap='viridis', figsize=(10,6))
plt.title('Performance Comparison Across Models', fontsize=14)
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 6. ROC Curves Plot
plt.figure(figsize=(10, 8))
for name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc = results[name]['AUC-ROC']
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
    
plt.plot([0, 1], [0, 1], 'k--') 
plt.title('ROC Curves Comparing All Models', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


best_model_name = "XGBoost (GPU)"
best_model = models[best_model_name]
joblib.dump(best_model, '../models/best_model.pkl')
print(f"\n🎯 Saved {best_model_name} as the final model for the application at 'models/best_model.pkl'!")
