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
            self.raw_file_path = self.params["data_ingestion"]["raw_data_path"]
            self.bucket_name = self.params["data_ingestion"]["bucket_name"]
            self.object_key = self.params["data_ingestion"]["object_key"]
            self.version_id = self.params["data_ingestion"]["version_id"]      
      def start_ingestion(self):
            try:
                  logging.info(" starting data loading process ")
                  logging.info("initiating connection to s3 ")
                  s3 = s3_connection.s3_operations(self.bucket_name , os.getenv('AWS_ACCESS_KEY_ID') ,os.getenv('AWS_SECRET_ACCESS_KEY') )
                  print(s3)
                  df = s3.fetch_file_from_s3(self.object_key, version_id = self.version_id)
                  logging.info( f" pulled dataframe of len - {len(df)}")
                  
                  final_df = self.validate_data(df)
                  save_DataFrame( self.raw_file_path , final_df )

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

      
      