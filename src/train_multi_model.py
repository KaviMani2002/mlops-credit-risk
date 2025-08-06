import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define constants
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
PREPROCESSOR_PATH = "data/processed/preprocessor.pkl"
EXPERIMENT_NAME = "credit_risk_multi_model"
FEATURES = ['loan_amt', 'int_rate', 'annual_inc', 'home_ownership', 'purpose', 'dti', 'term']
TARGET = 'loan_status'

# Load data
print("📥 Loading processed data...")
data = pd.read_csv(PROCESSED_DATA_PATH)

# Validate columns
missing = [col for col in FEATURES if col not in data.columns]
if missing:
    raise ValueError(f"⚠️ Preprocessing failed: columns are missing: {missing}. Available columns: {data.columns.tolist()}")

# Split features and target
X = data[FEATURES]
y = data[TARGET]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load preprocessor
print("🔧 Loading preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Transform features
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Define models
models = {
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "lr": LogisticRegression(max_iter=1000),
    "xgb": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Set experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Train and log each model
for model_name, model in models.items():
    print(f"\n🚀 Training model: {model_name}")

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train_transformed, y_train)
        y_pred = model.predict(X_test_transformed)

        # Evaluate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"📊 Metrics — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model", registered_model_name=f"credit_risk_{model_name}_model")

print("\n✅ All models trained and logged to MLflow.")

