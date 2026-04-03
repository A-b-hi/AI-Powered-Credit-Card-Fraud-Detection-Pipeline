# Phase 1: Exploratory Data Analysis (EDA)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

DATA_PATH = '../data/creditcard.csv'


# 1. Loading the Dataset
if not os.path.exists(DATA_PATH):
    print(f"⚠️ Dataset not found at {DATA_PATH}!")
    print("Please download 'creditcard.csv' from Kaggle and place it in the 'data' folder.")
    df = pd.DataFrame(np.random.randn(100, 31), columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class'])
    df['Class'] = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
else:
    df = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully!")

print(f"Dataset Shape: {df.shape}")
print("-" * 50)
print(df.info())

# 2. Checking for Missing Values
has_nan = df.isnull().sum().max()
if has_nan > 0:
    print(f"Found missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
else:
    print("No missing values found! The data is incredibly clean.")

# 3. Class Imbalance Analysis
class_counts = df['Class'].value_counts()
class_percentages = class_counts / len(df) * 100

print("\n--- Class Distribution ---")
print(f"Legit (0): {class_counts[0]} ({class_percentages[0]:.3f}%)")
print(f"Fraud (1): {class_counts[1]} ({class_percentages[1]:.3f}%)")

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Class', data=df, palette='viridis')
plt.title('Unbalanced Class Distribution\n(0: Legit | 1: Fraud)', fontsize=14)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + 0.3, p.get_height() + 100))
plt.show()


# 4. Transaction Amount distribution
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', alpha=0.5, label='Legit', kde=True)
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', alpha=0.5, label='Fraud', kde=True)
plt.title('Transaction Amount Distribution by Class', fontsize=14)
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Frequency')
plt.xlim(0, 2000) 
plt.legend()
plt.show()

# 5. Time Analysis

df['Time_in_Hours'] = df['Time'] / 3600

plt.figure(figsize=(12, 6))
sns.kdeplot(df[df['Class'] == 0]['Time_in_Hours'], label='Legit', fill=True, color='blue', alpha=0.3)
sns.kdeplot(df[df['Class'] == 1]['Time_in_Hours'], label='Fraud', fill=True, color='red', alpha=0.3)
plt.title('Density Plot of Transaction Time (in Hours)', fontsize=14)
plt.xlabel('Time (Hours from first transaction)')
plt.ylabel('Density')
plt.legend()
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, vmin=-1, vmax=1)
plt.title('Correlation Matrix (All Features)', fontsize=14)
plt.show()

corr_with_class = corr['Class'].sort_values(ascending=False)
print("\nTop Features POSITIVELY Correlated with Fraud:")
print(corr_with_class.head(6))
print("\nTop Features NEGATIVELY Correlated with Fraud:")
print(corr_with_class.tail(5))


print("EDA Complete! Now ready for Data Preprocessing and Model Training.")
