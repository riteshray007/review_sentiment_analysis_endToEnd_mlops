import sys
import os
from src.utils.main_utils import save_obj , read_yaml_file , load_DataFrame
from src.exception import MyException
from src.logger import logging
from sklearn.model_selection import RandomizedSearchCV , train_test_split

class ModelTraining:
      def __init__(self):
            self.params = read_yaml_file('params.yaml')
            self.processed_train_file = self.params["feature_engineering"]["vectorized_train_file_path"]
            self.processed_test_file = self.params["feature_engineering"]["vectorized_test_file_path"]
            self.model_file_path = self.params['model_training']["model_file_path"]
      
      def start_model_training(self):
            
            logging.info("inside ModelTraining start model training method ")
            train_data = load_DataFrame(self.processed_train_file)
            test_data = load_DataFrame(self.processed_test_file)
            logging.info("train test data loaded successfully ")
            
            
      
      
      

if __name__ == "__main__":
      trainer = ModelTraining()
      trainer.start_model_training()