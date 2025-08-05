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

# Config
FEATURES_PATH = '../data/processed/features.csv'
PREPROCESSOR_PATH = '../data/processed/preprocessor.pkl'
EXPERIMENT_NAME = "credit_risk_experiment"
TRACKING_URI = "http://172.17.0.1:5000"

# Setup MLflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Load data
df = pd.read_csv(FEATURES_PATH)
X = df.drop("loan_status", axis=1)
y = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)

# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ TEMP FIX: Inject fake class 1s for demo if y_train has only class 0
if len(y_train.unique()) == 1:
    print("⚠️ Only one class in y_train. Injecting synthetic 'Charged Off' samples for demo/testing.")
    X_synthetic = X_test.sample(10, random_state=42).copy()
    y_synthetic = pd.Series([1] * 10, index=X_synthetic.index)
    X_train = pd.concat([X_train, X_synthetic])
    y_train = pd.concat([y_train, y_synthetic])

feature_names = X.columns.tolist()
preprocessor = joblib.load(PREPROCESSOR_PATH)

models = {
    "rf_model": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced"),
    "logreg_model": LogisticRegression(max_iter=1000),
    "xgb_model": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

def save_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Train and log all models
for model_key, model in models.items():
    if len(set(y_train)) < 2:
        print(f"⚠️ Skipping {model_key}: y_train contains only one class ({set(y_train)}).")
        continue

    with mlflow.start_run(run_name=model_key):
        print(f"🚀 Training and logging: {model_key}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        mlflow.log_params(model.get_params())

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:1],
            registered_model_name=f"credit_risk_{model_key}"
        )

        joblib.dump(preprocessor, "preprocessor.pkl")
        mlflow.log_artifact("preprocessor.pkl")
        os.remove("preprocessor.pkl")

        with open("model_features.txt", "w") as f:
            f.write("\n".join(feature_names))
        mlflow.log_artifact("model_features.txt")
        os.remove("model_features.txt")

        cm_filename = f"{model_key}_confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, cm_filename)
        mlflow.log_artifact(cm_filename)
        os.remove(cm_filename)

        client = MlflowClient()
        latest_version = client.get_latest_versions(f"credit_risk_{model_key}", stages=["None"])[-1].version
        client.set_registered_model_tag(f"credit_risk_{model_key}", "feature_names", ",".join(feature_names))
        client.set_registered_model_tag(f"credit_risk_{model_key}", "feature_count", str(len(feature_names)))
        client.set_model_version_tag(f"credit_risk_{model_key}", latest_version, "run_id", mlflow.active_run().info.run_id)

        print(f"✅ Done: {model_key} | Accuracy: {acc:.4f}\n")

