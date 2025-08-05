import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# CONFIGURATION
# ------------------------------
DATA_PATH = "../data/processed/cleaned_data.csv"
PREPROCESSOR_PATH = "../data/processed/preprocessor.pkl"
FEATURE_COLS = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'purpose', 'emp_length']
TARGET_COL = "loan_status"

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv(DATA_PATH)

# Split features and target
X = df[FEATURE_COLS]
y = df[TARGET_COL]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# LOAD PREPROCESSOR AND TRANSFORM
# ------------------------------
preprocessor = joblib.load(PREPROCESSOR_PATH)
X_test_processed = preprocessor.transform(X_test)

# Optional: convert to DataFrame if needed
X_test_processed_df = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())

# ------------------------------
# CONNECT TO MLFLOW AND EVALUATE MODELS
# ------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# ✅ Fallback method for listing registered models
registered_models = [rm.name for rm in client.search_registered_models()]
print("🔍 Registered models found:", registered_models)

# Evaluate each model
for model_name in registered_models:
    print(f"\n🔍 Evaluating model: {model_name}")

    try:
        # Get latest version (None or Production stage)
        latest_versions = client.get_latest_versions(model_name, stages=["None", "Production"])
        if not latest_versions:
            print(f"⚠️ No versions found for {model_name}")
            continue

        version = latest_versions[0].version
        model_uri = f"models:/{model_name}/{version}"

        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        # Predict
        y_pred = model.predict(X_test_processed_df)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"✅ Accuracy for model '{model_name}' (v{version}): {acc:.4f}")
        print(report)

    except Exception as e:
        print(f"❌ Error evaluating model '{model_name}': {str(e)}")

