import json
import sys

import dagshub
import mlflow

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file


class ModelRegistry:
      def __init__(self):
            self.params = read_yaml_file("params.yaml")
            evaluation_config = self.params["model_evaluation"]
            registry_config = self.params["model_registry"]
            self.model_info_path = evaluation_config["model_info_path"]
            self.validation_report_path = evaluation_config["validation_report_path"]
            self.model_name = registry_config["model_name"]
            self.staging_alias = registry_config["staging_alias"]

            mlflow.set_tracking_uri(self.params["mlflow"]["tracking_uri"])
            dagshub.init(**self.params["mlflow"]["dagshub"])

      @staticmethod
      def load_json(file_path: str) -> dict:
            with open(file_path, "r", encoding="utf-8") as file:
                  return json.load(file)

      def load_validation_report(self, model_info: dict) -> dict:
            """Load the report and ensure it belongs to the model being registered."""
            report = self.load_json(self.validation_report_path)
            required_keys = {"run_id", "status", "checks"}
            missing_keys = required_keys - report.keys()
            if missing_keys:
                  raise ValueError(f"Validation report is missing keys: {sorted(missing_keys)}")
            if report["run_id"] != model_info["run_id"]:
                  raise ValueError(
                        "Validation report run_id does not match model_info run_id. "
                        "Run model evaluation again before registry."
                  )
            if report["status"] not in {"passed", "failed"}:
                  raise ValueError("Validation report status must be 'passed' or 'failed'.")
            if not report["checks"]:
                  raise ValueError("Validation report must contain at least one metric check.")

            checks_passed = all(check.get("passed") is True for check in report["checks"].values())
            expected_status = "passed" if checks_passed else "failed"
            if report["status"] != expected_status:
                  raise ValueError("Validation report status does not match its metric checks.")
            return report

      def register_model(self, model_info: dict, validation_report: dict):
            """Register every evaluated candidate; only passing candidates receive staging."""
            try:
                  model_version = mlflow.register_model(
                        model_uri=model_info["model_uri"],
                        name=self.model_name,
                  )
                  client = mlflow.tracking.MlflowClient()
                  validation_status = validation_report["status"]
                  client.set_model_version_tag(
                        name=self.model_name,
                        version=model_version.version,
                        key="validation_status",
                        value=validation_status,
                  )
                  client.set_model_version_tag(
                        name=self.model_name,
                        version=model_version.version,
                        key="validation_report_run_id",
                        value=validation_report["run_id"],
                  )
                                                            
                  if validation_status == "passed":
                        client.set_registered_model_alias(
                              name=self.model_name,
                              alias=self.staging_alias,
                              version=model_version.version,
                        )
                        logging.info(
                              "Registered model %s version %s and assigned alias %s",
                              self.model_name, model_version.version, self.staging_alias,
                        )
                  else:
                        logging.warning(
                              "Registered model %s version %s, but validation failed; %s was not changed.",
                              self.model_name, model_version.version, self.staging_alias,
                        )
                  return model_version
            except Exception as e:
                  raise MyException(e, sys) from e

      def start_model_registry(self):
            try:
                  model_info = self.load_json(self.model_info_path)
                  validation_report = self.load_validation_report(model_info)
                  model_version = self.register_model(model_info, validation_report)
                  logging.info("ModelRegistry completed for version %s", model_version.version)
                  return model_version
            except Exception as e:
                  logging.error("Model registry failed: %s", e)
                  raise MyException(e, sys) from e


if __name__ == "__main__":
      ModelRegistry().start_model_registry()