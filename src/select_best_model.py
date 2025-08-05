import os
import json
import joblib
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score

# Constants
DATA_PATH = './data/processed/cleaned_data.csv'
PREPROCESSOR_PATH = './data/processed/preprocessor.pkl'
BEST_MODEL_INFO_PATH = './data/processed/best_model_info.json'
EXPERIMENT_NAME = 'credit-risk-experiment'  # Make sure this matches your MLflow experiment

# 1. Load cleaned data
print("📥 Loading processed data...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Processed data not found at: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)

# 2. Load preprocessor
print("🔧 Loading preprocessor...")
if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(f"❌ Preprocessor not found at: {PREPROCESSOR_PATH}")
preprocessor = joblib.load(PREPROCESSOR_PATH)

# 3. Check required features
required_features = preprocessor.feature_names_in_
missing = set(required_features) - set(data.columns)
if missing:
    raise ValueError(f"⚠️ Preprocessing failed: columns are missing: {missing}")

# 4. Convert loan_status to binary target
print("🎯 Converting loan_status to target variable...")
data['target'] = data['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
data = data.dropna(subset=['target'])  # Drop unknown labels

# 5. Prepare features and target
X = data[required_features]
y = data['target']

# 6. Transform features using preprocessor
print("🔄 Transforming features...")
X_transformed = preprocessor.transform(X)

# 7. Initialize MLflow and load runs
print("🔍 Loading MLflow experiment runs...")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(f"❌ No experiment named '{EXPERIMENT_NAME}' found in MLflow.")

experiment_id = experiment.experiment_id
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=[experiment_id], order_by=["metrics.accuracy DESC"])

if not runs:
    raise ValueError("❌ No runs found in the MLflow experiment.")

# 8. Evaluate each model and find the best
print("🔍 Evaluating models...")
best_model = None
best_accuracy = 0
best_model_uri = None
best_run_id = None

for run in runs:
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        y_pred = model.predict(X_transformed)
        acc = accuracy_score(y, y_pred)
        print(f"✅ Run {run_id} - Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_uri = model_uri
            best_run_id = run_id

    except Exception as e:
        print(f"⚠️ Skipping run {run_id} due to error: {e}")

if best_model is None:
    raise RuntimeError("❌ No model could be loaded or passed evaluation.")

# 9. Save best model info
print(f"🏆 Best Model Run ID: {best_run_id} | Accuracy: {best_accuracy:.4f}")
best_model_info = {
    "run_id": best_run_id,
    "model_uri": best_model_uri,
    "accuracy": best_accuracy
}

os.makedirs(os.path.dirname(BEST_MODEL_INFO_PATH), exist_ok=True)
with open(BEST_MODEL_INFO_PATH, 'w') as f:
    json.dump(best_model_info, f, indent=2)

print(f"✅ Best model info saved to {BEST_MODEL_INFO_PATH}")

