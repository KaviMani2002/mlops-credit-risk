# src/feature_engineering.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_DATA_PATH = "data/processed/processed_data.csv"
PREPROCESSOR_PATH = "data/processed/preprocessor.pkl"
X_PROCESSED_PATH = "data/processed/X_processed.pkl"
Y_PATH = "data/processed/y.pkl"

def load_data(path):
    print("📥 Loading cleaned data...")
    return pd.read_csv(path)

def build_preprocessor():
    numeric_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
    categorical_features = ['term', 'purpose']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    return preprocessor

def run_feature_engineering():
    df = load_data(INPUT_PATH)
    print(f"✅ Input shape: {df.shape}")

    expected_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'term', 'purpose', 'loan_status']
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    df['loan_status'] = df['loan_status'].astype(int)
    y = df['loan_status']
    X = df.drop(columns=['loan_status'])

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    column_names = (
        preprocessor.named_transformers_['num'].get_feature_names_out(['loan_amnt', 'int_rate', 'annual_inc', 'dti']).tolist() +
        preprocessor.named_transformers_['cat'].get_feature_names_out(['term', 'purpose']).tolist()
    )

    X_final = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
        columns=column_names
    )

    X_final['loan_status'] = y.values

    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)

    X_final.to_csv(OUTPUT_DATA_PATH, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(X_final.drop(columns=['loan_status']), X_PROCESSED_PATH)
    joblib.dump(y, Y_PATH)

    print("✅ Feature engineering complete.")
    print(f"📦 Processed data saved to: {OUTPUT_DATA_PATH}")
    print(f"🧠 Preprocessor saved to: {PREPROCESSOR_PATH}")
    print(f"🧮 X_processed.pkl and y.pkl saved to: {X_PROCESSED_PATH}, {Y_PATH}")

if __name__ == "__main__":
    run_feature_engineering()

