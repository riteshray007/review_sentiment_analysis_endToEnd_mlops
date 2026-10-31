import os
from pathlib import Path

project_name = 'src'
note_book='notebooks'
flask_app = "flask-app"
k8s = "k8s"

list_of_files = [
      f"{project_name}/__init__.py",
      f"{project_name}/components/__init__.py",
      f"{project_name}/components/data_ingestion.py",
      f"{project_name}/components/data_preprocessing.py",
      f"{project_name}/components/feature_engineering.py",
      f"{project_name}/components/model_training.py",
      f"{project_name}/components/model_evaluation.py",
      f"{project_name}/components/model_registry.py",
      f"{project_name}/s3_services/__init__.py",
      f"{project_name}/s3_services/s3_connection.py",
      f"{project_name}/entity/config_entity.py",
      f"{project_name}/logger/__init__.py",
      f"{project_name}/exception/__init__.py",
      f"{project_name}/utils/__init__.py",
      f"{project_name}/utils/main_utils.py",
      "requirements.txt",
      "pyproject.toml",
      f"{note_book}/exp1.ipynb",
      f"{note_book}/exp2_bow_vs_tfidf.py",
      f"{note_book}/exp3_lor_bow_hp.py",
      "tests/test_model.py",
      "tests/test_flask_app.py",
      "scripts/promote_model.py",
      f"{flask_app}/app.py",
      f"{flask_app}/requirements.txt",
      f"{flask_app}/templates/index.html",
      "local_s3/__init__.py",
      "params.yaml",
      ".dockerignore",
      "Dockerfile",
      f"{k8s}/grafana-values.yaml",
      f"{k8s}/flask-deployment.yaml",
      f"{k8s}/prometheus-values.yaml",
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
            
            
      