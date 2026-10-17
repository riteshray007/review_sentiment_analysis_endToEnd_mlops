import sys
import os
from src.utils.main_utils import save_obj , read_yaml_file , load_DataFrame , write_yaml_file
from src.exception import MyException
from src.logger import logging
from sklearn.model_selection import RandomizedSearchCV , train_test_split
from scipy.sparse import load_npz
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score , f1_score , precision_score , recall_score

class ModelTraining:
      def __init__(self):
            self.params = read_yaml_file('params.yaml')
            self.best_params_file_path = self.params['feature_engineering']['best_parameters_file_path']
            self.x_train_file = self.params["feature_engineering"]["vectorized_x_train_file_path"]
            self.x_test_file = self.params["feature_engineering"]["vectorized_x_test_file_path"]
            self.y_train_file = self.params["feature_engineering"]["vectorized_y_train_file_path"]
            self.y_test_file = self.params["feature_engineering"]["vectorized_y_test_file_path"]
            self.model_file_path = self.params['model_training']["model_file_path"]
            self.model_metrics_path = self.params['model_training']["model_metrics_path"]
      
      def start_model_training(self):
            
            logging.info("inside ModelTraining start model training method ")
            x_train = load_npz(self.x_train_file)
            x_test = load_npz(self.x_test_file)
            y_train = load_DataFrame(self.y_train_file)
            y_test = load_DataFrame(self.y_test_file)
            # logging.info(f"y_train - {y_train.head()}")
            # logging.info(f"y_test - {y_test.head()}")
            logging.info("train test data loaded successfully ")
            
            logging.info("obtaining best paramaeters for model from best_params.yaml ")
            param = self.get_best_params(
                  self.best_params_file_path
            )
            
            logging.info("starting model training ")
            model = self.model_training(param , x_train , x_test , y_train ,  y_test)
            
            logging.info(f"saving trained model at- {self.model_file_path}")
            save_obj(self.model_file_path , model )
            
      def model_training(self , param , x_train , x_test , y_train ,  y_test ):
            
            model = SGDClassifier(**param)
            model.fit(x_train , y_train)
            y_pred = model.predict(x_test)
            
            # metrics = {
            #       "test_accuracy" : accuracy_score(y_test , y_pred) ,
            #       "test_recall" : recall_score(y_test , y_pred) ,
            #       "test_precision" : precision_score(y_test , y_pred) ,
            #       "test_f1_score" : f1_score(y_test , y_pred) 
            # }
      
            # write_yaml_file(self.model_metrics_path ,  metrics)   
            
            return model                 
      
      
      def get_best_params(self , file_path ):
            param = read_yaml_file(file_path)
            return param['model_training']
            

if __name__ == "__main__":
      trainer = ModelTraining()
      trainer.start_model_training()