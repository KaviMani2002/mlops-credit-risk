import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

warnings.filterwarnings("ignore")

# ================================
# Setup
# ================================
DATA_PATH = "data/processed/cleaned_data.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_tracking_uri("http://20.106.177.129:5000")
mlflow.set_experiment("credit_risk_experiment")

# ================================
# Load Data
# ================================
print("📥 Loading cleaned data...")
df = pd.read_csv(DATA_PATH)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# Models to Train
# ================================
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# ================================
# Train & Log Models
# ================================
for model_name, model in models.items():
    print(f"🚀 Training: {model_name}")

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"✅ {model_name} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Log params, metrics, model
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, model_name.lower())

        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_filename = os.path.join(ARTIFACT_DIR, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_filename)
        plt.close()

        # Log confusion matrix plot to MLflow
        mlflow.log_artifact(cm_filename)

print("✅ All models trained and logged to MLflow.")

