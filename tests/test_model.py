import unittest
import os
import mlflow
from mlflow import MlflowClient
from src.utils.main_utils import read_yaml_file
import pandas as pd

class TestModel(unittest.TestCase):
      
      @classmethod
      def setUpClass(cls):
            # 1. Verify MLflow Credentials
            mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
            mlflow_pass =  os.getenv('MLFLOW_TRACKING_PASSWORD')
            
            if not mlflow_username or not mlflow_pass :
                  raise EnvironmentError('missing environment variable mlflow username/password')
            
            # 2. Read configs and set MLflow tracking URI
            cls.params = read_yaml_file('params.yaml')
            mlflow.set_tracking_uri(cls.params['mlflow']['tracking_uri'])
            
            # 3. Get Model Name and Staging Alias from params.yaml
            cls.model_name = cls.params['model_registry']['model_name']
            cls.staging_alias = cls.params['model_registry']['staging_alias']

            # 4. Fetch the version metadata using the staging alias
            client = MlflowClient()
            model_version_detail = client.get_model_version_by_alias(
                  name = cls.model_name,
                  alias = cls.staging_alias
            )
            cls.latest_model_version = model_version_detail.version
            
            # 5. Load the model directly using the @alias URI 
            cls.model_uri = f"models:/{cls.model_name}@{cls.staging_alias}"
            cls.model = mlflow.pyfunc.load_model(cls.model_uri)
            
            test_data_path = cls.params['data_preprocessing']['test_file_path']
            cls.test_df = pd.read_csv(test_data_path)
            
      def test_model_loaded_properly(self):
            self.assertIsNotNone(self.model, "Model failed to load from MLflow")
            self.assertIsNotNone(self.latest_model_version, "Model version could not be resolved")
            
      def test_model_signature(self):
            input_text = "Brilliant direction and stunning visual effects! Even at nearly three hours, the pacing never falters. I was glued to my seat the entire time"
            input_df = pd.DataFrame({"review" : [input_text]})
            
            prediction = self.model.predict(input_df)
            
            #since shape returns (row_count , col_count ) shape[0] = row_count & shape[1] = col_count 
            self.assertEqual(input_df.shape[0], 1, "Input dataframe should have 1 row")
            self.assertEqual(input_df.shape[1], 1, "Input dataframe should have 1 column")
            
            # 4. Assert output shape matches input length (1 sample in -> 1 prediction out)
            self.assertEqual(len(prediction), len(input_df), "Output length should match input row count")

            # 5. Assert output dimension (should be a 1D array of predictions)
            self.assertEqual(len(prediction.shape), 1, "Prediction should be a 1D array")
      
            # 6. Assert predicted class is a valid sentiment label (0: negative, 1: positive)
            self.assertIn(int(prediction[0]), [0, 1], "Prediction should be either 0 or 1")
      
      def test_model_performance(self):
            
            X_test = self.test_df[['review']]
            y_true = self.test_df['sentiment']
            
            y_pred = self.model.predict(X_test)
            
            from sklearn.metrics import accuracy_score , f1_score , precision_score , recall_score
            acc = accuracy_score( y_true , y_pred )
            f1 = f1_score(y_true , y_pred)
            prec = precision_score(y_true , y_pred)
            rec = recall_score(y_true , y_pred)
            
            thresholds = self.params['model_evaluation']['validation_thresholds']
            
            self.assertGreaterEqual(acc , thresholds['accuracy'],f"Accuracy {acc:.4f} below threshold {thresholds['accuracy']}")
            self.assertGreaterEqual(f1 , thresholds['f1_score'],f"F1-score {f1:.4f} below threshold {thresholds['f1_score']}")
            self.assertGreaterEqual(prec , thresholds['precision'],f"Precision {prec:.4f} below threshold {thresholds['precision']}")
            self.assertGreaterEqual(rec , thresholds['recall'], f"Recall {rec:.4f} below threshold {thresholds['recall']}")
                 
      

if __name__ == "__main__":
      unittest.main()