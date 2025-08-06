import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = "data/processed/cleaned_data.csv"
PREPROCESSOR_PATH = "data/processed/preprocessor.pkl"
MLFLOW_TRACKING_DIR = "mlruns"  # <-- Use relative path for CI/CD compatibility

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"file://{os.path.abspath(MLFLOW_TRACKING_DIR)}")

# Load data
print("📥 Loading processed data...")
data = pd.read_csv(DATA_PATH)

# Feature columns
FEATURES = [
    "loan_amt",
    "int_rate",
    "annual_inc",
    "home_ownership",
    "purpose",
    "dti",
    "term"
]
TARGET = "loan_status"

X = data[FEATURES]
y = data[TARGET]

# Encode target if needed
if y.dtype == 'O':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to train
models = {
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgb": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "lr": LogisticRegression(max_iter=1000)
}

# Start training
for name, model in models.items():
    print(f"\n🚀 Training model: {name}")
    with mlflow.start_run(run_name=f"{name}_run", nested=True):

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"📊 Metrics — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        # Log parameters and metrics
        mlflow.log_param("model_type", name)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save metrics plot
        plot_path = f"outputs/metrics_{name}.png"
        os.makedirs("outputs", exist_ok=True)
        plt.figure(figsize=(5, 3))
        sns.barplot(x=["Accuracy", "Precision", "Recall", "F1"], y=[acc, prec, rec, f1])
        plt.title(f"Metrics for {name.upper()}")
        plt.ylim(0, 1)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        # Register model (optional)
        mlflow.set_experiment("credit_risk_models")
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name=f"credit_risk_{name}_model"
        )

print("\n✅ Training and logging completed.")

