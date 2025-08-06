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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# -------------------- Setup --------------------
mlflow.set_tracking_uri("http://20.106.177.129:5000")
mlflow.set_experiment("credit_risk_experiment")

# -------------------- Load Data --------------------
print("📥 Loading cleaned data...")
data_path = "data/processed/cleaned_data.csv"
df = pd.read_csv(data_path)

# -------------------- Features & Target --------------------
target = "loan_status"
features = ['loan_amnt', 'term', 'int_rate', 'emp_length', 'annual_inc', 'dti', 'purpose']

X = df[features]
y = df[target]

# -------------------- Column Types --------------------
categorical_cols = ['term', 'emp_length', 'purpose']
numerical_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']

# -------------------- Preprocessor --------------------
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Save preprocessor (optional)
preprocessor_path = "artifacts/preprocessor.pkl"
os.makedirs("artifacts", exist_ok=True)
joblib.dump(preprocessor, preprocessor_path)
print(f"💾 Preprocessor saved to: {preprocessor_path}")

# -------------------- Models --------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# -------------------- Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Train & Log Loop --------------------
for model_name, model_instance in models.items():
    print(f"\n🚀 Training: {model_name}")

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model_instance)
    ])

    with mlflow.start_run(run_name=model_name):
        # Set tags
        mlflow.set_tag("model_name", model_name)

        # Log model params
        mlflow.log_params(model_instance.get_params())

        # Train
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"✅ {model_name} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = f"artifacts/{model_name}_conf_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Log model and register it
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name="credit-risk-model"
        )

print("\n✅✅ All models trained and logged to MLflow!")

