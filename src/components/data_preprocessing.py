import sys
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file , load_DataFrame  , save_DataFrame , clean_review
import pandas as pd
import numpy as np
import re
import html


from sklearn.model_selection import train_test_split
import os



class DataPreProcessing:
      def __init__(self):
            self.params = read_yaml_file("params.yaml")
            self.raw_file_path = self.params['data_ingestion']['raw_data_path']
            self.test_size = self.params['feature_engineering']['test_size']
            self.random_state = self.params['feature_engineering']['random_state']
            self.processed_train_file_path = self.params['data_preprocessing']['train_file_path']
            self.processed_test_file_path = self.params['data_preprocessing']['test_file_path']
      
      def normalize_text(self ,df):
            try:
                  logging.info("started validate and normalize labels method ")
                  df = self.validate_and_normalize_labels(df)

                  logging.info("started clean review method ")
                  df["review"] = df["review"].apply(clean_review)

                  # Remove reviews that become empty after cleaning
                  df = df[df["review"].str.strip().ne("")].copy()

                  return df
            
            except Exception as e:
                  raise MyException(e , sys ) from e

      
      def validate_and_normalize_labels(self ,df):
            required_columns = {"review", "sentiment"}
            missing_columns = required_columns - set(df.columns)

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            df = df.dropna(subset=["review", "sentiment"]).copy()

            df["sentiment"] = (
                df["sentiment"].astype(str).str.strip().str.lower()
            )

            label_mapping = {"positive": 1, "negative": 0}
            df["sentiment"] = df["sentiment"].map(label_mapping)

            if df["sentiment"].isna().any():
                raise ValueError(
                    "Unexpected sentiment labels found in the dataset."
                )

            df["sentiment"] = df["sentiment"].astype(int)
            df["review"] = df["review"].astype(str)
            df = df[df["review"].str.strip().ne("")]

            return df


      
      def start_data_preprocessing(self):
            
            try:
                  
                  logging.info('inside start_data_preprocessing method of DataPreProcessing class ')

                  # if os.path.exists(self.processed_train_file_path) and  os.path.exists(self.processed_test_file_path) :
                  #       logging.info('preprocessed file alredy exist moving to the next component')
                  #       return
                  df = load_DataFrame(self.raw_file_path)

                  logging.info("started text normalization ")
                  final_df = self.normalize_text(df)

                  y = final_df['sentiment']
                  train , test  = train_test_split(
                      final_df,
                      test_size= self.test_size,
                      random_state=self.random_state,
                      stratify=y,
                  )

                  logging.info("train test split complete ")

                  save_DataFrame( self.processed_train_file_path  , train)
                  save_DataFrame( self.processed_test_file_path  , test)
                  logging.info("data preprocessing is completed and exiting preprocessing component ")
                  
            except Exception as e:
                  logging.error(f"error - {e}")
                  raise MyException(e , sys ) from e
      
if __name__ == "__main__":
      preprocessing = DataPreProcessing()
      logging.info("entered DataPreProcessing class  , started data pre processing ")
      preprocessing.start_data_preprocessing()
      