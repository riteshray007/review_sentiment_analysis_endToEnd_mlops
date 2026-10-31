from flask import Flask , render_template , request
import mlflow
import mlflow.sklearn
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
# import dagshub
from src.utils.main_utils import clean_review , read_yaml_file
from src.logger import logging
from src.exception import MyException
import sys

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

app = Flask(__name__)

param = read_yaml_file("params.yaml")
mlflow_tracking_uri = param["mlflow"]["tracking_uri"]
logging.info(f'tracking uri loaded - {mlflow_tracking_uri}')
# dagshub_params = param['mlflow']['dagshub']
# logging.info(f'dagshub params loaded - {dagshub_params}')

mlflow.set_tracking_uri(mlflow_tracking_uri)
# dagshub.init(**dagshub_params)

# from prometheus_client import CollectorRegistry

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

model_name = param['model_registry']['model_name']
model_alias = param['model_registry']['production_alias']

def load_staging_model():
      """Load the model currently assigned to the configured MLflow staging alias."""
      model_uri = f"models:/{model_name}@{model_alias}"
      try:
            logging.info("Loading registered model from MLflow: %s", model_uri)
            return mlflow.sklearn.load_model(model_uri)
      except Exception:
            logging.exception("Failed to load registered model from MLflow: %s", model_uri)
            raise

latest_ml_model = load_staging_model()


# @app.route('/')
# def home():
#     REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
#     start_time = time.time()
#     response = render_template("index.html", result=None)
#     REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    # return response

@app.route('/healthz', methods=['GET'])
def health_check():
    return "healthy", 200

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}
      
      
@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["POST"])
def home():
    if request.method == "POST":
        REQUEST_COUNT.labels(method=request.method, endpoint=request.path).inc()
        start_time = time.time()
        
        global latest_ml_model
        if not latest_ml_model:
            latest_ml_model = load_staging_model()
            
        text = request.form.get("text", "")
        df = pd.DataFrame({"review": [text]})
        result = latest_ml_model.predict(df)[0]

        PREDICTION_COUNT.labels(prediction=str(result)).inc()
        REQUEST_LATENCY.labels(endpoint=request.path).observe(time.time() - start_time)

        # Return JSON for frontend fetch and CI/CD tests
        sentiment = "Positive" if result == 1 else "Negative"
        return {"result": int(result), "sentiment": sentiment}, 200

    # GET request
    REQUEST_COUNT.labels(method=request.method, endpoint=request.path).inc()
    start_time = time.time()
    response = render_template("index.html")
    REQUEST_LATENCY.labels(endpoint=request.path).observe(time.time() - start_time)
    return response


if __name__ == "__main__":
      app.run(debug=True , host="0.0.0.0" , port = 5000 )