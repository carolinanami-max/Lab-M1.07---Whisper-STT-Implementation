import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Step 1: Load the data
# Assuming the file is named 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Explore the dataset
print("=" * 50)
print("STEP 1: DATA EXPLORATION")
print("=" * 50)

# Basic information
print(f"Dataset shape: {df.shape}")
print(f"\nDataset columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values in each column:")
print(df.isnull().sum())

# Check unique values in categorical columns
print("\nUnique values in categorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {df[col].unique()[:10]}")  # Show first 10 unique values

# Target distribution
print(f"\nTarget variable (Churn) distribution:")
print(df['Churn'].value_counts())
print(f"\nChurn percentage: {df['Churn'].value_counts(normalize=True) * 100}")

# Visualize target distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Churn', data=df)
plt.title('Distribution of Churn')
plt.xlabel('Churn')
plt.ylabel('Count')

# Add percentages on bars
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.show()

# Visualize some key features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Tenure distribution
axes[0, 0].hist(df['tenure'], bins=30, edgecolor='black')
axes[0, 0].set_title('Tenure Distribution')
axes[0, 0].set_xlabel('Tenure (months)')
axes[0, 0].set_ylabel('Count')

# 2. Monthly Charges distribution
axes[0, 1].hist(df['MonthlyCharges'], bins=30, edgecolor='black', color='orange')
axes[0, 1].set_title('Monthly Charges Distribution')
axes[0, 1].set_xlabel('Monthly Charges ($)')
axes[0, 1].set_ylabel('Count')

# 3. Churn by Contract type
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
contract_churn.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Churn Rate by Contract Type')
axes[1, 0].set_xlabel('Contract Type')
axes[1, 0].set_ylabel('Percentage')
axes[1, 0].legend(title='Churn')

# 4. Churn by Internet Service
internet_churn = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack()
internet_churn.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Churn Rate by Internet Service')
axes[1, 1].set_xlabel('Internet Service')
axes[1, 1].set_ylabel('Percentage')
axes[1, 1].legend(title='Churn')

plt.tight_layout()
plt.savefig('key_features_visualization.png')
plt.show()

# Additional visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Churn by Payment Method
payment_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
payment_churn.plot(kind='bar', ax=axes[0])
axes[0].set_title('Churn Rate by Payment Method')
axes[0].set_xlabel('Payment Method')
axes[0].set_ylabel('Percentage')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(title='Churn')

# Churn by Senior Citizen status
senior_churn = df.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).unstack()
senior_churn.index = ['Non-Senior', 'Senior']
senior_churn.plot(kind='bar', ax=axes[1])
axes[1].set_title('Churn Rate by Senior Citizen Status')
axes[1].set_xlabel('Senior Citizen')
axes[1].set_ylabel('Percentage')
axes[1].legend(title='Churn')

plt.tight_layout()
plt.savefig('additional_features_visualization.png')
plt.show()
