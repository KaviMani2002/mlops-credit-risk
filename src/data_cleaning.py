# src/data_cleaning.py

import os
import pandas as pd

# ✅ Updated path to use ingested data instead of raw
INPUT_PATH = "../data/ingested/ingested_data.csv"
OUTPUT_PATH = "../data/processed/cleaned_data.csv"

# Selected useful columns (can update later if needed)
SELECTED_COLUMNS = [
    "loan_amnt", "term", "int_rate", "emp_length", "annual_inc",
    "dti", "purpose", "delinq_2yrs", "loan_status"
]

# Mapping target variable
STATUS_MAP = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1
}

def clean_data():
    print(f"🔄 Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"🔍 Original shape: {df.shape}")

    # Filter required columns
    df = df[SELECTED_COLUMNS]
    print(f"📦 Columns selected: {df.columns.tolist()}")

    # Drop rows with missing values
    df = df.dropna(subset=["loan_status"])
    df = df.dropna()

    # Binary encode target
    df = df[df["loan_status"].isin(STATUS_MAP.keys())]
    df["loan_status"] = df["loan_status"].map(STATUS_MAP)
    print("✅ Class distribution:", df["loan_status"].value_counts().to_dict())

    # Clean up term
    df["term"] = df["term"].str.strip()

    # Save cleaned data
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Cleaned data saved to: {OUTPUT_PATH}")
    print(f"🧾 Final shape: {df.shape}")

if __name__ == "__main__":
    clean_data()

