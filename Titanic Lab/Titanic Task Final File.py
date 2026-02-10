# File: titanic_model_final.py
# STEP 1: Import Libraries
print("=" * 50)
print("STEP 1: Importing Libraries")
print("=" * 50)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

print("✓ All libraries imported successfully!")

# STEP 2: Import Dataset
print("\n" + "=" * 50)
print("STEP 2: Importing Titanic Dataset")
print("=" * 50)
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
print(f"✓ Dataset loaded!")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"   Columns: {list(df.columns)}")

# STEP 3: Preprocess Data
print("\n" + "=" * 50)
print("STEP 3: Preprocessing Data")
print("=" * 50)
# Select only the columns we need and drop missing values
df_clean = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
print(f"✓ After cleaning: {df_clean.shape[0]} rows remaining")

# Convert categorical to numerical
df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})

# Create features (X) and target (y)
X = df_clean[['Pclass', 'Sex', 'Age', 'Fare']]
y = df_clean['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Data split:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features standardized (mean=0, std=1)")

# STEP 4: Build Logistic Regression Model
print("\n" + "=" * 50)
print("STEP 4: Building Logistic Regression Model")
print("=" * 50)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully!")

# STEP 5: Make Predictions
print("\n" + "=" * 50)
print("STEP 5: Making Predictions")
print("=" * 50)
y_pred = model.predict(X_test_scaled)
print(f"✓ Made {len(y_pred)} predictions")
print(f"   Sample predictions vs actual:")
for i in range(5):
    print(f"   Prediction: {y_pred[i]}, Actual: {y_test.iloc[i]}")

# STEP 6: Evaluate with F1 Score
print("\n" + "=" * 50)
print("STEP 6: Evaluation with F1 Score")
print("=" * 50)
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

print("\n" + "=" * 50)
print("INTERPRETATION")
print("=" * 50)
print("F1 Score ranges from 0 to 1:")
print("0.0 = Model performs terribly")
print("0.5 = Model is guessing randomly")
print("1.0 = Perfect predictions")
print(f"\nYour model score: {f1:.4f}")
if f1 > 0.7:
    print("Great job! This is a good model for a first attempt.")
elif f1 > 0.5:
    print("Good start! The model is learning patterns.")
else:
    print("Keep learning! Every model is a step forward.")

print("\n✓ All steps completed successfully!")