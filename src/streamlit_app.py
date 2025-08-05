# src/streamlit_app.py

import streamlit as st
import mlflow
import mlflow.pyfunc
import pandas as pd
import tempfile
import joblib
import logging
from logging.handlers import RotatingFileHandler
from mlflow.tracking import MlflowClient
import sys
import os

# --- Logging Setup ---
log_dir = "/opt/mlops-credit-risk-demo/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "streamlit_app.log")

handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[handler]
)

# --- Local Project Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Monitoring ---
from monitoring.monitoring import track_prediction, start_prometheus_server

logging.info("👉 Starting Prometheus metrics server")
start_prometheus_server()
logging.info("✅ Prometheus metrics server started")

# --- Tracing Setup (OpenTelemetry + Jaeger) ---
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor

resource = Resource(attributes={"service.name": "credit-risk-streamlit"})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",  # or 'jaeger' if Docker
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
provider.add_span_processor(span_processor)

RequestsInstrumentor().instrument()
tracer = trace.get_tracer(__name__)
logging.info("✅ Tracing initialized with Jaeger exporter")

# --- Constants ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

RAW_FEATURES = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'home_ownership', 'purpose', 'term']
CATEGORICAL_COLS = ['home_ownership', 'purpose', 'term']
NUMERICAL_COLS = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']


# --- Load Model and Preprocessor ---
@st.cache_resource(show_spinner="Loading model and preprocessor...")
def load_model_artifacts(model_name, version):
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.pyfunc.load_model(model_uri)

    run_id = client.get_model_version(model_name, str(version)).run_id
    local_dir = tempfile.mkdtemp()

    # Load preprocessor
    preprocessor_path = client.download_artifacts(run_id, "preprocessor.pkl", local_dir)
    preprocessor = joblib.load(preprocessor_path)

    # Load feature names
    try:
        feature_path = client.download_artifacts(run_id, "model_features.txt", local_dir)
        with open(feature_path, "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
    except Exception:
        feature_names = list(preprocessor.get_feature_names_out())

    return model, preprocessor, feature_names


# --- Prediction Logic with Tracing and Monitoring ---
@track_prediction
def make_prediction(model, preprocessor, df, feature_names):
    with tracer.start_as_current_span("single_prediction"):
        X_array = preprocessor.transform(df)
        X_df = pd.DataFrame(X_array, columns=feature_names)
        return model.predict(X_df)[0]

@track_prediction
def make_batch_prediction(model, preprocessor, df, feature_names):
    with tracer.start_as_current_span("batch_prediction"):
        X_array = preprocessor.transform(df)
        X_df = pd.DataFrame(X_array, columns=feature_names)
        return model.predict(X_df)


# --- Streamlit UI ---
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
st.title("📊 Credit Risk Prediction App")

# --- Sidebar: Model Selection ---
st.sidebar.header("🧠 Select Model")
registered_models = client.search_registered_models()
model_names = sorted([m.name for m in registered_models])
selected_model = st.sidebar.selectbox("Registered Model", model_names)

if selected_model:
    versions_info = client.search_model_versions(f"name='{selected_model}'")
    version_list = sorted([int(v.version) for v in versions_info], reverse=True)
    selected_version = st.sidebar.selectbox("Model Version", version_list)

    model, preprocessor, feature_names = load_model_artifacts(selected_model, selected_version)
    st.sidebar.success(f"✅ Loaded: {selected_model} v{selected_version}")
    logging.info(f"✅ Loaded model: {selected_model} v{selected_version}")

    # --- Tabs ---
    tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Batch Prediction"])

    # --- Tab 1: Single Prediction ---
    with tab1:
        st.subheader("Enter Customer Information")

        input_data = {}
        for col in NUMERICAL_COLS:
            input_data[col] = st.number_input(f"{col}", value=0.0)

        input_data['home_ownership'] = st.selectbox("home_ownership", ['MORTGAGE', 'RENT', 'OWN', 'OTHER', 'NONE', 'ANY'])
        input_data['purpose'] = st.selectbox("purpose", [
            'debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business',
            'car', 'wedding', 'house', 'medical', 'moving', 'vacation', 'educational',
            'renewable_energy', 'other'
        ])
        input_data['term'] = st.selectbox("term", ['36 months', '60 months'])

        if st.button("Predict", type="primary"):
            try:
                df = pd.DataFrame([input_data])
                pred = make_prediction(model, preprocessor, df, feature_names)
                st.success(f"Prediction: {'⚠️ High Risk (1)' if pred == 1 else '✅ Low Risk (0)'}")
                logging.info(f"Single prediction made: {input_data} → {pred}")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                logging.error(f"Prediction error: {e}")

    # --- Tab 2: Batch Prediction ---
    with tab2:
        st.subheader("Upload CSV for Batch Prediction")
        uploaded_file = st.file_uploader("Upload a CSV file with the 7 input features", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                missing = set(RAW_FEATURES) - set(df.columns)
                if missing:
                    st.error(f"Missing columns in uploaded file: {missing}")
                    logging.warning(f"Uploaded file missing columns: {missing}")
                else:
                    preds = make_batch_prediction(model, preprocessor, df, feature_names)
                    df["prediction"] = preds
                    df["prediction"] = df["prediction"].map({1: "1 (High Risk)", 0: "0 (Low Risk)"})
                    st.dataframe(df.head())

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
                    logging.info(f"Batch prediction completed: {len(df)} records")

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                logging.error(f"Batch prediction error: {e}")

