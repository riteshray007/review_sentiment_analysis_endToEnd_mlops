from mlflow import MlflowClient
from src.utils.main_utils import read_yaml_file
import os
import sys
import mlflow
from src.logger import logging
from src.exception import MyException

def promote_staging_to_production():
      """
      fetches the current staging model 
      and promotes it to production stage
      """
      try:
            mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
            mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')
            
            if not mlflow_username or not mlflow_password:
                  raise EnvironmentError(" missing environment MLFLOW_TRACKING_USERNAME/MLFLOW_TRACKING_PASSWORD")
            
            params = read_yaml_file('params.yaml')
            mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])    
            
            registry_config = params['model_registry']
            model_name = registry_config['model_name']
            staging_alias = registry_config['staging_alias']
            production_alias = registry_config['production_alias']
            pre_prod_alias = registry_config['pre_production']            
            
            client = MlflowClient()
            
            staging_model = client.get_model_version_by_alias(
                  name = model_name,
                  alias = staging_alias
            )
            staging_version = staging_model.version
            logging.info(f"Found staging model: Version {staging_version} (Run ID: {staging_model.run_id})")
            
            # 4. Check what is currently in production (for audit logging)
            try:
                  current_prod_model = client.get_model_version_by_alias(
                      name=model_name,
                      alias=production_alias
                  )
                  current_prod_version = current_prod_model.version
                  logging.info(f"Current model in '{production_alias}': Version {current_prod_model.version}")
                
                  if current_prod_version != staging_version:
                      logging.info(
                          f"Assigning alias '{pre_prod_alias}' to Version {current_prod_version} for rollback capability..."
                      )
                      client.set_registered_model_alias(
                          name=model_name,
                          alias=pre_prod_alias,
                          version=current_prod_version
                      )
                  else:
                      logging.info(f"Version {staging_version} is already in production. Skipping rollback alias shift.")                                
            except Exception:
                logging.info(f"No existing model found with alias '{production_alias}'. First time promotion.")
            
            # 5. Reassign the production alias to the staging version
            logging.info(f"Promoting Version {staging_version} to alias '{production_alias}'...")
            client.set_registered_model_alias(
                name=model_name,
                alias=production_alias,
                version=staging_version
            )
            
            # 6. Optional: Add a tracking tag for audit history
            client.set_model_version_tag(
                name=model_name,
                version=staging_version,
                key="promoted_to_production",
                value="true"
            )
            
            logging.info(f"Successfully promoted model '{model_name}' version {staging_version} to '@{production_alias}'")

      except Exception as e:
            logging.error(f" failed to promote model:{e}")
            raise MyException(e,sys) from e      

if __name__ == "__main__":
      promote_staging_to_production()