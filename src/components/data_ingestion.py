import pandas as pd
from src.utils.main_utils import read_yaml_file , load_DataFrame , save_DataFrame
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import MyException
from src.s3_services import s3_connection
import sys
import os


class DataIngestion:
      
      def __init__(self):
            self.params = read_yaml_file("params.yaml")
            self.raw_file_path = self.params['data_ingestion']['raw_data_path']
      
      def start_ingestion(self):
            try:
                  logging.info(" starting data loading process ")

                  if not os.path.exists(self.raw_file_path):
                        logging.info("initiating connection to s3 ")
                        s3 = s3_connection.s3_operations("sentiment-analysis-review-data", os.getenv('AWS_ACCESS_KEY_ID') ,os.getenv('AWS_SECRET_ACCESS_KEY') )
                        df = s3.fetch_file_from_s3("IMDB Dataset.csv")
                        logging.info( f" pulled dataframe of len - {len(df)}")
                        
                        final_df = self.validate_data(df)
                        save_DataFrame( self.raw_file_path , final_df )
                  else:
                        logging.info("raw data file already exists")     
                        logging.info("exiting data ingestion ")     
            except Exception as e:
                  logging.error(f"error= {e}")
                  raise MyException(e,sys) from e
      
      def validate_data(self , df:pd.DataFrame ) -> pd.DataFrame:
            logging.info(" entered inside data validation method ")
            logging.info(f"data frame length before preprocessing - {len(df)}" )
            final_df = df[df['sentiment'].isin(['negative' , 'positive'])]
            logging.info(f"data frame length after preprocessing - {len(final_df)}" )
            return final_df
      
      
if __name__ == "__main__":
      logging.info("starting Data Ingestion Component")
      ingestion = DataIngestion()
      ingestion.start_ingestion()

      
      