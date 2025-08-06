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
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# ----------------- Configuration -------------------
PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
PREPROCESSOR_PATH = 'data/processed/preprocessor.pkl'
EXPERIMENT_NAME = "credit_risk_experiment"
TRACKING_URI = "http://20.106.177.129:5000"  # Update if needed

# Ensure temp folder exists
TEMP_DIR = "temp_artifacts"
os.makedirs(TEMP_DIR, exist_ok=True)

# ----------------- MLflow Setup ---------------------
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
os.environ["MLFLOW_ARTIFACT_URI"] = "./mlruns"

# ----------------- Load Data ------------------------
print("📥 Loading processed data...")
df = pd.read_csv(PROCESSED_DATA_PATH)
X = df.drop("loan_status", axis=1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
feature_names = X.columns.tolist()

# ----------------- Load Preprocessor ----------------
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ----------------- Models to Train ------------------
models = {
    "rf": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced"),
    "logreg": LogisticRegression(max_iter=1000),
    "xgb": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# ----------------- Helper: Save Confusion Matrix ----
def save_confusion_matrix(y_true, y_pred, filepath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# ----------------- Train and Log --------------------
for model_key, model in models.items():
    with mlflow.start_run(run_name=model_key):
        print(f"\n🚀 Training model: {model_key}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"📊 Metrics — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Log model parameters
        mlflow.log_params(model.get_params())

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        registered_model_name = f"credit_risk_{model_key}_model"

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:1],
            signature=signature,
            registered_model_name=registered_model_name
        )

        # Log preprocessor
        preprocessor_path = os.path.join(TEMP_DIR, f"{model_key}_preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(preprocessor_path)
        os.remove(preprocessor_path)

        # Log feature list
        feature_path = os.path.join(TEMP_DIR, f"{model_key}_model_features.txt")
        with open(feature_path, "w") as f:
            f.write("\n".join(feature_names))
        mlflow.log_artifact(feature_path)
        os.remove(feature_path)

        # Log confusion matrix
        cm_path = os.path.join(TEMP_DIR, f"{model_key}_confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Add model tags
        client = MlflowClient()
        versions = client.get_latest_versions(registered_model_name, stages=["None"])
        if versions:
            latest_version = versions[-1].version
            client.set_registered_model_tag(registered_model_name, "feature_names", ",".join(feature_names))
            client.set_registered_model_tag(registered_model_name, "feature_count", str(len(feature_names)))
            client.set_model_version_tag(registered_model_name, latest_version, "run_id", mlflow.active_run().info.run_id)

        print(f"✅ Completed: {model_key} | Logged to MLflow\n")

