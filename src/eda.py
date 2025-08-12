# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Paths
DATA_PATH = "../data/processed/cleaned_data.csv"
PLOT_DIR = "../data/eda_plots/"

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

print("📊 Loading cleaned data for EDA...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Data shape: {df.shape}")

# Convert target column to string for visualization
df["loan_status"] = df["loan_status"].map({0: "No Risk", 1: "Risk"})

# 1. Target variable distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="loan_status", data=df)
plt.title("Class Distribution")
plt.savefig(f"{PLOT_DIR}/class_distribution.png")
plt.close()

# 2. Histograms for numerical features
num_cols = ["loan_amnt", "int_rate", "annual_inc", "dti", "delinq_2yrs"]
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"{PLOT_DIR}/{col}_histogram.png")
    plt.close()

# 3. Boxplots grouped by risk
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="loan_status", y=col, data=df)
    plt.title(f"{col} vs Loan Status")
    plt.savefig(f"{PLOT_DIR}/{col}_vs_loan_status.png")
    plt.close()

# 4. Categorical plot for 'purpose'
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="purpose", hue="loan_status")
plt.title("Purpose vs Loan Status")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/purpose_vs_loan_status.png")
plt.close()

# 5. Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig(f"{PLOT_DIR}/correlation_heatmap.png")
plt.close()

print(f"📈 EDA complete. Plots saved to: {PLOT_DIR}")

