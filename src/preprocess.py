import pandas as pd
import numpy as np
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Paths
RAW_DATA_PATH = "../data/raw/accepted_2007_to_2018Q4.csv"
FEATURES_OUTPUT_PATH = "../data/processed/features.csv"
PREPROCESSOR_OUTPUT_PATH = "../data/processed/preprocessor.pkl"

# Final 7 input features + target
SELECTED_COLUMNS = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "home_ownership",
    "purpose",
    "dti",
    "term",
    "loan_status"
]

def load_raw_data():
    print("📥 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    df = df[SELECTED_COLUMNS]
    df = df.dropna()
    
    # Clean 'int_rate' if it's a percentage string
    if df["int_rate"].dtype == "object":
        df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float)

    # Clean 'term' - remove leading/trailing spaces
    df["term"] = df["term"].str.strip()

    return df

def build_pipeline(numeric_features, categorical_features):
    print("🔧 Building preprocessing pipeline...")
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor

def preprocess_and_save():
    df = load_raw_data()

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    numeric_features = ["loan_amnt", "int_rate", "annual_inc", "dti"]
    categorical_features = ["home_ownership", "purpose", "term"]

    preprocessor = build_pipeline(numeric_features, categorical_features)

    print("⚙️ Fitting and transforming data...")
    X_processed = preprocessor.fit_transform(X)

    cat_feature_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)

    processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
    processed_df["loan_status"] = y.reset_index(drop=True)

    print("💾 Saving processed features and preprocessor...")
    os.makedirs(os.path.dirname(FEATURES_OUTPUT_PATH), exist_ok=True)
    processed_df.to_csv(FEATURES_OUTPUT_PATH, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_OUTPUT_PATH)

    print(f"✅ Done: Processed features saved to: {FEATURES_OUTPUT_PATH}")
    print(f"✅ Done: Preprocessor saved to: {PREPROCESSOR_OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_and_save()

