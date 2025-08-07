import os
import logging
import mlflow
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup MLflow client
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Define model names to evaluate
model_names = [
    "credit_risk_rf_model",
    "credit_risk_logreg_model",
    "credit_risk_xgb_model",
    "credit_risk_model"
]

# Metric to compare
METRIC_NAME = "precision"

def get_latest_model_version(model_name):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        # Sort by version number descending and return the latest
        return sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    except Exception as e:
        logger.error(f"⚠️ Error fetching model version for {model_name}: {e}")
        return None

def get_metric_from_run(run_id, metric_name):
    try:
        metric_history = client.get_metric_history(run_id, metric_name)
        if metric_history:
            return metric_history[-1].value
        else:
            return 0.0
    except Exception as e:
        logger.error(f"⚠️ Failed to get metric '{metric_name}' from run {run_id}: {e}")
        return 0.0

def promote_best_model():
    best_model_name = None
    best_model_version = None
    best_metric = -1

    for model_name in model_names:
        version_info = get_latest_model_version(model_name)
        if not version_info:
            logger.warning(f"⚠️ No versions found for model: {model_name}")
            continue

        run_id = version_info.run_id
        version = version_info.version
        precision = get_metric_from_run(run_id, METRIC_NAME)

        logger.info(f"🔍 {model_name} version {version} has precision: {precision}")

        if precision > best_metric:
            best_metric = precision
            best_model_name = model_name
            best_model_version = version

    if best_model_name and best_model_version:
        try:
            client.transition_model_version_stage(
                name=best_model_name,
                version=best_model_version,
                stage="Staging",
                archive_existing_versions=True,
            )
            logger.info(f"✅ Promoted '{best_model_name}' version {best_model_version} to Staging with precision: {best_metric}")
        except Exception as e:
            logger.error(f"❌ Failed to promote '{best_model_name}': {e}")
    else:
        logger.warning("⚠️ No valid model found for promotion.")

if __name__ == "__main__":
    promote_best_model()

