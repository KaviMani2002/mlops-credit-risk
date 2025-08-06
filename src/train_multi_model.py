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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ---------------------- Configuration ----------------------
EXPERIMENT_NAME = "credit_risk_experiment"
MLFLOW_TRACKING_URI = "http://20.106.177.129:5000"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------------- Set MLflow ----------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------- Load Data ----------------------
print("📥 Loading cleaned data...")
df = pd.read_csv("data/processed/cleaned_data.csv")

# ---------------------- Define Features ----------------------
target = 'loan_status'
categorical_features = ['term', 'purpose']
numerical_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']

X = df[categorical_features + numerical_features]
y = df[target]

# ---------------------- Split Data ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------- Preprocessor ----------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ---------------------- Model Configs ----------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

# ---------------------- Training & MLflow Logging ----------------------
for model_name, model in models.items():
    print(f"🚀 Training: {model_name}")
    with mlflow.start_run(run_name=model_name):
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"✅ {model_name} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name="credit-risk-model"
        )

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_filename = os.path.join(ARTIFACT_DIR, f"{model_name}_conf_matrix.png")
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename, artifact_path="confusion_matrices")
        plt.close()

print("✅ All models trained and logged to MLflow.")

