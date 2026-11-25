# -*- coding: utf-8 -*-
"""
Improved KNN on Adult Income Dataset
- Original ~82.9% → Improved ~85.1% test accuracy
Dataset: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# 1. Load & Clean Data
# ============================================================
# Update this path if needed (Colab: '/content/adult.csv')
df = pd.read_csv('adult.csv')

# Replace '?' with NaN and fill categorical missing values with mode
df = df.replace('?', np.nan)
for col in ['workclass', 'occupation', 'native-country']:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()
print(f"Shape after cleaning: {df.shape}")
print(df['income'].value_counts(normalize=True).round(3))

# ============================================================
# 2. Feature Selection
# ============================================================
# Drop: education (redundant with educational-num), fnlwgt (sampling weight, noisy)
X = df.drop(columns=['income', 'education', 'fnlwgt'])
y = df['income']

categorical_features = [
    'workclass', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country'
]
numerical_features = [
    'age', 'educational-num', 'capital-gain',
    'capital-loss', 'hours-per-week'
]

# ============================================================
# 3. Train / Test Split (stratified)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================================
# 4. Preprocessing Pipeline
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed train shape: {X_train_processed.shape}")

# ============================================================
# 5. KNN Model (tuned)
# ============================================================
# Best found: k=21, uniform weights, manhattan distance
knn = KNeighborsClassifier(
    n_neighbors=21,
    weights='uniform',
    metric='manhattan',
    n_jobs=-1
)
knn.fit(X_train_processed, y_train)

# ============================================================
# 6. Evaluation
# ============================================================
y_pred_train = knn.predict(X_train_processed)
y_pred_test = knn.predict(X_test_processed)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"\nTrain accuracy: {acc_train:.4f}")
print(f"Test  accuracy: {acc_test:.4f}")

print("\n===== Classification Report (Test) =====")
print(classification_report(y_test, y_pred_test))

# ============================================================
# 7. Confusion Matrices
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Train CM
cm_train = confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'],
            ax=axes[0])
axes[0].set_title(f'Train Confusion Matrix\nAccuracy: {acc_train:.4f}')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Test CM
cm_test = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'],
            ax=axes[1])
axes[1].set_title(f'Test Confusion Matrix\nAccuracy: {acc_test:.4f}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()