import os
import sys
import pandas as pd
from src.utils.main_utils import read_yaml_file , load_DataFrame , write_yaml_file , save_obj , save_DataFrame
from src.logger import logging
from src.exception import MyException
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from scipy.stats import loguniform
import yaml
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV ,StratifiedKFold
import numpy as np


class FeatureEngineering:
      def __init__(self):
            self.params = read_yaml_file('params.yaml')
            self.random_state = self.params['feature_engineering']['random_state']
            self.processed_train_file = self.params['data_preprocessing']['train_file_path']
            self.processed_test_file = self.params['data_preprocessing']['test_file_path']
            self.best_params_file_path = self.params['feature_engineering']['best_parameters_file_path']
            self.vectorized_train_file_path = self.params['feature_engineering']['vectorized_x_train_file_path']
            self.vectorized_y_train_file_path = self.params['feature_engineering']['vectorized_y_train_file_path']
            self.vectorized_test_file_path = self.params['feature_engineering']['vectorized_x_test_file_path']
            self.vectorized_y_test_file_path = self.params['feature_engineering']['vectorized_y_test_file_path']
            self.preprocessing_obj_file_path = self.params['feature_engineering']['preprocessed_model_file_path']
            self.random_cv_params = self.params['randomized_cv_search']
            
      def start_feature_engineering(self):
            
            try:
                  train_df:pd.DataFrame = load_DataFrame(self.processed_train_file)
                  test_df:pd.DataFrame = load_DataFrame(self.processed_test_file)
                  logging.info("successfully loaded train test dataframe ")

                  print(train_df["review"].dtype)
                  print(train_df["review"].isna().sum())
                  print(train_df["review"].map(type).value_counts())
                  print(train_df["review"].shape)

                  param = self.get_best_params(
                        train_df['review'],
                        train_df['sentiment']
                  ) 

                  logging.info(f" best params for tfidf vectorizer is - {param}")

                  tfidf_params = param['feature_engineering']

                  logging.info("starting tfidf vectorization ")
                  vectorizer = TfidfVectorizer(**tfidf_params)

                  X_train = vectorizer.fit_transform(train_df["review"])
                  y_train = train_df['sentiment']
                  X_test = vectorizer.transform(test_df["review"])
                  y_test = test_df['sentiment']
                  logging.info('vectorizer fitting complete')

                  os.makedirs(os.path.dirname(self.vectorized_train_file_path) , exist_ok=True)
                  os.makedirs(os.path.dirname(self.vectorized_test_file_path) , exist_ok=True)
                  save_npz( self.vectorized_train_file_path , X_train)
                  save_npz( self.vectorized_test_file_path , X_test)
                  logging.info("saved vectorized train test target values successfully in vectors ")
                  
                  save_DataFrame(self.vectorized_y_train_file_path , y_train)
                  save_DataFrame(self.vectorized_y_test_file_path , y_test)
                  logging.info("saved target columns in data/vectorized directory ")

                  save_obj(self.preprocessing_obj_file_path , vectorizer )
                  logging.info(f"saved preprocessor.pkl file at {self.preprocessing_obj_file_path}")
                  
            except Exception as e:
                  raise MyException(e , sys ) from e 

                
      def get_best_params(self , X_train , y_train ):
            try:
                  logging.info("entered inside get_best_params method ")
                  if os.path.exists(self.best_params_file_path):
                        logging.info("Loading saved best parameters")
                        
                        with open(self.best_params_file_path, "r") as file:
                              params = yaml.safe_load(file) or {}
                              
                        if (
                            "feature_engineering" in params
                            and "model_training" in params
                        ):
                            
                            params["feature_engineering"]["ngram_range"] = tuple(params["feature_engineering"]["ngram_range"])
                            return params
                      
                        logging.warning("Saved parameters are incomplete. Running randomized search again.")
                        
                  # 2. Build a pipeline
                  
                  pipeline = Pipeline([
                      ("tfidf", TfidfVectorizer()),
                      ("classifier", SGDClassifier(
                          random_state=self.random_state
                      )),
                  ])
                  
                  param_distributions = {
                      # TF-IDF parameters
                      "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                      "tfidf__min_df": [1, 2, 3, 5],
                      "tfidf__max_df": [0.9, 0.95, 1.0],
                      "tfidf__sublinear_tf": [True, False],
                      "tfidf__max_features": [30000, 50000, None],

                      # Additional TF-IDF parameters
                      "tfidf__norm": ["l1", "l2"],
                      "tfidf__binary": [True, False],
                      "tfidf__use_idf": [True, False],

                      # SGDClassifier parameters
                      "classifier__loss": ["hinge", "log_loss", "modified_huber"],
                      "classifier__alpha": loguniform(1e-6, 1e-3),
                      "classifier__penalty": ["l2", "l1", "elasticnet"],
                      "classifier__class_weight": [None, "balanced"],
                      "classifier__average": [True, False],

                      # Additional convergence parameters
                      "classifier__tol": [1e-3, 1e-4, 1e-5],
                      "classifier__max_iter": [1000, 2000, 3000],

                  }

                  cv = StratifiedKFold(
                      n_splits=5,
                      shuffle=True,
                      random_state=self.random_state
                  )

                  search = RandomizedSearchCV(
                      estimator=pipeline,
                      param_distributions=param_distributions,
                      scoring={
                          "f1_macro": "f1_macro",
                          "accuracy": "accuracy",
                          "precision_macro": "precision_macro",
                          "recall_macro": "recall_macro",
                      },
                      cv=cv,
                      **self.random_cv_params
                  )
                  
                  logging.info("starting randomized cv search ")
                  # X_train_text must contain cleaned text, not TF-IDF features
                  search.fit(X_train , y_train)
                  
                  logging.info('randomized cv search is over and fitting is complete')
                  # 4. Separate parameters by pipeline step
                  best = search.best_params_
                  logging.info(f'best pramas for randomized cv search is - {best} ')
                  
                  feature_params = {
                      key.removeprefix("tfidf__"): value
                      for key, value in best.items()
                      if key.startswith("tfidf__")
                  }
                  
                  model_params = {
                      key.removeprefix("classifier__"): value
                      for key, value in best.items()
                      if key.startswith("classifier__")
                  }
                  
                  result = {
                      "feature_engineering": feature_params,
                      "model_training": model_params,
                  }
                  
                  yaml_result = self.convert_numpy_types(result)
                  
                  # 5. Save the YAML file
                  write_yaml_file(self.best_params_file_path,yaml_result)
                  logging.info(f"best params saved in path - {self.best_params_file_path}")    
                  logging.info(f"Best CV score: {search.best_score_:.4f}")

                  return result

            except Exception as e:
                  raise MyException(e,sys ) from e
      

      def convert_numpy_types(self ,params):
          logging.info(f"enterd convert_numpy_types method ")
          if isinstance(params, dict):
              return {
                  key: self.convert_numpy_types(value)
                  for key, value in params.items()
              }

          if isinstance(params, tuple):
              return [self.convert_numpy_types(v) for v in params]

          if isinstance(params, list):
              return [self.convert_numpy_types(v) for v in params]

          if isinstance(params, np.generic):
              return params.item()

          return params

if __name__ == "__main__":
      feature_eng = FeatureEngineering()
      # print(feature_eng.random_cv_params)
      feature_eng.start_feature_engineering()