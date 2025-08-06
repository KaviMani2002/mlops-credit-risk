import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------- Configuration -------------------
PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
PREPROCESSOR_PATH = 'data/processed/preprocessor.pkl'
EXPERIMENT_NAME = "credit_risk_experiment"
TRACKING_URI = "http://20.106.177.129:5000"

# ----------------- MLflow Setup ---------------------
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

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

# ----------------- Train and Log --------------------
for model_key, model in models.items():
    with mlflow.start_run(run_name=model_key):
        print(f"\n🚀 Training model: {model_key}")

        # Fit
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"📊 Metrics — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        # Log params
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Log model and register under credit-risk-model
        #mlflow.sklearn.log_model(
         #   sk_model=model,
          #  artifact_path="model",
           # registered_model_name="credit-risk-model"
        #)

        print(f"✅ Completed: {model_key} | Logged to MLflow\n")

