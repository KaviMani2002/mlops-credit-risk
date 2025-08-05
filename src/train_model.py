import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
FEATURES_PATH = '../data/processed/features.csv'
PREPROCESSOR_PATH = '../data/processed/preprocessor.pkl'
EXPERIMENT_NAME = "credit_risk_experiment"
REGISTERED_MODEL_NAME = "credit_risk_model"

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns.tolist()

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

def train_and_log_model():
    print("✅ [1] Setting MLflow URI and experiment...")
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("✅ [2] Starting run...")
    with mlflow.start_run() as run:
        print("✅ [3] Loading data...")
        (X_train, X_test, y_train, y_test), feature_names = load_data()

        print("✅ [4] Loading preprocessor...")
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        print("✅ [5] Training model...")
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "random_state": 42,
            "class_weight": "balanced"
        }
        mlflow.log_params(params)

        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        print("✅ [6] Predicting and calculating metrics...")
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("✅ [7] Logging metrics...")
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        print("✅ [8] Inferring signature and input example...")
        signature = infer_signature(X_train, clf.predict(X_train))
        input_example = X_train.iloc[:1]

        print("✅ [9] Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME
        )

        print("✅ [10] Logging preprocessor...")
        joblib.dump(preprocessor, "preprocessor.pkl")
        mlflow.log_artifact("preprocessor.pkl")
        os.remove("preprocessor.pkl")

        print("✅ [11] Logging model features...")
        with open("model_features.txt", "w") as f:
            f.write("\n".join(feature_names))
        mlflow.log_artifact("model_features.txt")
        os.remove("model_features.txt")

        print("✅ [12] Tagging registered model...")
        client = MlflowClient()
        model_name = REGISTERED_MODEL_NAME
        latest_version = client.get_latest_versions(model_name, stages=["None"])[-1].version

        client.set_registered_model_tag(model_name, "feature_count", str(len(feature_names)))
        client.set_registered_model_tag(model_name, "feature_names", ",".join(feature_names))
        client.set_model_version_tag(model_name, latest_version, "run_id", run.info.run_id)

        print("✅ [13] Logging confusion matrix plot...")
        cm_filename = "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, cm_filename)
        mlflow.log_artifact(cm_filename)
        os.remove(cm_filename)

        print(f"✅ [14] Done. Model logged to MLflow — Accuracy: {acc:.4f}")
        print(f"🧪 View run at: http://localhost:5000/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run.info.run_id}")

if __name__ == "__main__":
    train_and_log_model()

