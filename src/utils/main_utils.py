import html
import os
import re
import sys
import contractions
import emoji

import numpy as np
import pandas as pd
import dill
import yaml
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging


def read_yaml_file(file_path:str)->dict:
      try:
            with open(file_path,'rb') as yaml_file:
                  return yaml.safe_load(yaml_file)
      except Exception as e:
            raise MyException(e,sys) from e


def write_yaml_file(file_path:str , content:object , replace:bool=False)->None:
      try:
            if replace:
                  if os.path.exists(file_path):
                        os.remove(file_path)
            os.makedirs(os.path.dirname(file_path) , exist_ok=True )
            with open(file_path,'w') as file:
                  yaml.safe_dump(content,file)
      except Exception as e :
            raise MyException(e ,sys ) from e
      

def load_object(file_path:str) -> object:
    """
    Returns model/object from project directory.
    file_path: str location of file to load
    return: Model/Obj
    """
    try:
          with open(file_path,'rb') as file:
                obj = dill.load(file)
          return obj
    except Exception as e:
          raise MyException(e,sys) from e


def save_numpy_array_data(file_path:str,array=np.array):
      """
      Save numpy array data to file
      file_path: str location of file to save
      array: np.array data to save      
      """
      try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path,exist_ok=True)
            with open(file_path , 'wb') as file_obj:
                  np.save(file_obj,array)
      except Exception as e:
            raise MyException(e,sys) from e

def load_numpy_array_data(file_path:str)->np.array:
      """
      load numpy array data from file
      file_path: str location of file to load
      return: np.array data loaded
      """
          
      try:
            with open(file_path , 'rb') as obj:
                  return np.load(obj)
      except Exception as e:
            raise MyException(e,sys) from e
      
def load_DataFrame(file_path:str)->DataFrame:
      try:
            return pd.read_csv(file_path)
      except Exception as e:
            raise MyException(e,sys) from e
      
def save_DataFrame(file_path:str , df:DataFrame ):
      try:
            dir_name = os.path.dirname(file_path)
            if not os.path.exists( dir_name ):
                  os.makedirs( dir_name , exist_ok=True)
            df.to_csv(file_path,index=False)
            
      except Exception as e:
            raise MyException(e,sys) 
      
def save_obj(file_path:str , obj:object)->None:
      logging.info('entered save obj method in utils')
      
      try:
            os.makedirs(os.path.dirname(file_path),exist_ok=True)
            with open(file_path,'wb') as file_obj:
                  dill.dump(obj , file_obj)
            logging.info('exited the save obj method ')
      except Exception as e:
            raise MyException(e,sys) from e

            
def clean_text_batch(values):
      if isinstance(values, pd.DataFrame):
          texts = values.iloc[:, 0].tolist()
      elif isinstance(values, pd.Series):
          texts = values.tolist()
      else:
          values = np.asarray(values, dtype=object)
          if values.ndim == 2:
              if values.shape[1] != 1:
                  raise ValueError("Expected one review-text column.")
              texts = values[:, 0].tolist()
          else:
              texts = values.tolist()
      return [clean_review(text) for text in texts]
      
def clean_review( text):
      """Conservative text normalization for sentiment analysis."""
      # 1. Handle missing values safely
      if pd.isna(text):
        return ""
  
      text = str(text)
      # 2. Decode HTML entities, e.g. &amp; -> &
      text = html.unescape(text)            
      # 3. Remove HTML tags common in IMDB reviews
      text = re.sub(r"<[^>]+>", " ", text)
      # 4. Replace URLs with a placeholder
      text = re.sub(
          r"(https?://\S+|www\.\S+)",
          " URLTOKEN ",
          text,
          flags=re.IGNORECASE
      )
      # 5. Normalize common emoticons before punctuation removal
      emoticons = {
          r":-?\)": "POSITIVEEMOTION",
          r";-?\)": " POSITIVEEMOTION ",
          r":-?\(": " NEGATIVEEMOTION ",
          r"<3": " HEARTTOKEN ",
          r":-?D": " POSITIVEEMOTION "
      }
      
      for pattern, replacement in emoticons.items():
          text = re.sub(
              pattern,
              replacement,
              text,
              flags=re.IGNORECASE
          )
      # 6. Convert emojis into descriptive words
      text = emoji.demojize(text, delimiters=(" ", " "))
      # 7. Expand contractions: didn't -> did not
      text = contractions.fix(text)
      # 8. Normalize case
      text = text.lower()
      # 9. Remove apostrophes and punctuation without
      # joining adjacent words together
      text = re.sub(r"[^\w\s]", " ", text)
      # 10. Keep digits initially; normalize whitespace
      text = re.sub(r"\s+", " ", text).strip()
      return text