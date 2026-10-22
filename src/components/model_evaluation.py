import mlflow
import mlflow.sklearn
import dagshub
import sys
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

from src.logger import logging
from src.exception import MyException
import os
import json
from src.utils.main_utils import load_DataFrame , load_object , read_yaml_file , write_yaml_file 
from sklearn.metrics import accuracy_score , recall_score , precision_score , f1_score , roc_auc_score


class modelEvaluation:
      def __init__(self):
            self.params = read_yaml_file("params.yaml")
            self.model_eval_metrics_path = self.params['model_evaluation']['model_eval_metrics_path']
            self.model_info_path = self.params['model_evaluation']['model_info_path']
            self.validation_report_path = self.params['model_evaluation']['validation_report_path']
            self.validation_thresholds = self.params['model_evaluation']['validation_thresholds']
            self.trained_model_path = self.params['model_training']['model_file_path']
            self.x_test_file_path = self.params['feature_engineering']['vectorized_x_test_file_path'] 
            self.y_test_file_path = self.params['feature_engineering']['vectorized_y_test_file_path'] 
            self.best_params = self.params['feature_engineering']['best_parameters_file_path']
            mlflow.set_tracking_uri(self.params['mlflow']['tracking_uri'])
            dagshub.init( **self.params['mlflow']['dagshub'] )

      def eval_metrics(self , model , x_test , y_test):
            y_pred = model.predict(x_test)
            y_pred_prob = model.predict_proba(x_test)[:,1]
            
            metrics = {
                  "accuracy" : accuracy_score(y_test , y_pred),
                  "precision" : precision_score(y_test , y_pred),
                  "recall" : recall_score(y_test , y_pred),
                  "f1_score" : f1_score(y_test , y_pred ),
                  "auc_score" : roc_auc_score(y_test , y_pred_prob)
            }
            
            logging.info("model's metrics calculated  ")
            
            return metrics

      def build_validation_report(self, metrics, run_id):
            """Evaluate configured metric floors and return a registry-ready report."""
            checks = {}
            for metric_name, threshold in self.validation_thresholds.items():
                  if metric_name not in metrics:
                        raise ValueError(f"Threshold configured for unknown metric: {metric_name}")

                  actual = float(metrics[metric_name])
                  checks[metric_name] = {
                        "actual": actual,
                        "minimum": float(threshold),
                        "passed": actual >= float(threshold),
                  }

            return {
                  "run_id": run_id,
                  "metrics": {name: float(value) for name, value in metrics.items()},
                  "thresholds": self.validation_thresholds,
                  "checks": checks,
                  "status": "passed" if all(check["passed"] for check in checks.values()) else "failed",
            }

      def save_validation_report(self, report):
            """Persist the report consumed by the model registry gate."""
            os.makedirs(os.path.dirname(self.validation_report_path), exist_ok=True)
            with open(self.validation_report_path, "w", encoding="utf-8") as file:
                  json.dump(report, file, indent=2)
            

      def experiment_tracking(self , model , x_test , y_test):
            try:
                  mlflow.set_experiment("my-dvc-experiments")
                  with mlflow.start_run() as run:
                        metrics = self.eval_metrics(model , x_test , y_test)

                        write_yaml_file(self.model_eval_metrics_path , metrics)
                        logging.info("model metrics saved successfully")

                        validation_report = self.build_validation_report(metrics, run.info.run_id)
                        self.save_validation_report(validation_report)
                        logging.info("Validation status: %s . saved successfully ", validation_report["status"] )
                        
                        mlflow.log_metrics(metrics)
                        
                        model_params = read_yaml_file(self.best_params)
                        model_params = model_params['model_training']
                        
                        mlflow.log_params(model_params)
                        
                        logged_model_info = mlflow.sklearn.log_model(
                              sk_model=model,
                              name="model"
                        )
                        
                        self.save_model_info(
                              run.info.run_id,
                              "model", 
                              self.model_info_path,
                              logged_model_info
                        )

                        mlflow.log_artifact(self.model_eval_metrics_path)
                        mlflow.log_artifact(self.validation_report_path)
                              
            except Exception as e:
                  raise MyException(e,sys) from e   
            
      def save_model_info(self , run_id , model_name , model_path , logged_model_info ):
            """Save the model run ID and path to a JSON file."""
            try:
                  os.makedirs(os.path.dirname(model_path) , exist_ok=True)
                  model_info = {'run_id': run_id,
                                'model_path': model_name,
                                "model_id":logged_model_info.model_id,
                                "model_name":logged_model_info.name,
                                "model_uri":logged_model_info.model_uri
                              }
                  with open(model_path, 'w') as file:
                      json.dump(model_info, file, indent=4)
                  logging.debug('Model info saved to %s', model_path)
            except Exception as e:
                logging.error('Error occurred while saving the model info: %s', e)
                raise   
            
      def start_model_evaluation(self):
            try:
                  logging.info("inside start model evaluation method of ModelEvaluation ")
                  x_test = load_npz(self.x_test_file_path)
                  y_test = load_DataFrame(self.y_test_file_path)
                  print(x_test.shape)
                  print(y_test.shape)
                  
                  logging.info("loading saved model for evaluation")
                  model = load_object(self.trained_model_path)
                  self.experiment_tracking(model , x_test , y_test)
                  
            except Exception as e:
                  raise MyException(e , sys ) from e

if __name__ == "__main__":
      model_eval = modelEvaluation()
      model_eval.start_model_evaluation()
      # print(model_eval.params)
