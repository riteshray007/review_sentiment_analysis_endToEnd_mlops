import pandas as pd
from src.utils.main_utils import read_yaml_file , load_DataFrame , save_DataFrame
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import MyException
import sys


class DataIngestion():
      
      def __init__(self):
            self.params = read_yaml_file("params.yaml")
            self.test_size = self.params['data_ingestion']['test_size']
            self.file_path = self.params['data_ingestion']['file_path']
            self.random_state = self.params['data_ingestion']['random_state']
            self.train_file_path = self.params['data_ingestion']['raw_train_file_path']
            self.test_file_path = self.params['data_ingestion']['raw_test_file_path']
      
      def start_ingestion(self):
            try:
                logging.info(" starting data loading process ")
                df  = load_DataFrame(self.file_path)
                print(df.head())
    
                final_df = self.preprocess_data(df)    
                train_df , test_df = train_test_split(final_df , test_size=self.test_size , random_state=self.random_state  )
                logging.info(f"saving train df at {self.train_file_path}")
                save_DataFrame(  self.train_file_path , train_df)
                logging.info(f"saving test df at {self.test_file_path}")
                save_DataFrame(self.test_file_path , test_df)
                logging.info("saved dfs sucessfully")
            except Exception as e:
                  logging.error(f"error= {e}")
                  raise MyException(e,sys) from e
      
      def preprocess_data(self , df:pd.DataFrame ) -> pd.DataFrame:
            logging.info(" entered inside preprocessing data method ")
            logging.info(f"data frame length before preprocessing - {len(df)}" )
            final_df = df[df['sentiment'].isin(['negative' , 'positive'])]
            logging.info(f"data frame length after preprocessing - {len(final_df)}" )
            return final_df
      
      
if __name__ == "__main__":
      ingestion = DataIngestion()
      ingestion.start_ingestion()

      
      