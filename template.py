import os
from pathlib import Path

project_name = 'src'
note_book='notebooks'

list_of_files = [
      f"{project_name}/__init__.py",
      f"{project_name}/components/__init__.py",
      f"{project_name}/components/data_ingestion.py",
      f"{project_name}/components/data_preprocessing.py",
      f"{project_name}/entity/config_entity.py",
      f"{project_name}/logger/__init__.py",
      f"{project_name}/exception/__init__.py",
      f"{project_name}/utils/__init__.py",
      f"{project_name}/utils/main_utils.py",
      "requirements.txt",
      "demo.py",
      "pyproject.toml",
      f"{note_book}/exp1.ipynb",
      f"{note_book}/exp2_bow_vs_tfidf.py",
      f"{note_book}/exp3_lor_bow_hp.py",
      "local_s3/__init__.py",
      "params.yaml"
]

for filepath in list_of_files:
      filepath = Path(filepath)
      filedir , filename = os.path.split(filepath)
      if filedir!='':
            os.makedirs(filedir , exist_ok=True)
      if ( not os.path.exists(filepath) ):
            filepath.touch(exist_ok=True)
      else:
            print(f"file is already present at: {filepath}")
            
            
      