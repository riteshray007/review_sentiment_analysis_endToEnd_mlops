import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
import contractions
import emoji
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================

CONFIG = {
    "data_path": "notebooks/IMDB.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/riteshray007/review_sentiment_analysis_endToEnd_mlops.mlflow",
    "dagshub_repo_owner": "riteshray007",
    "dagshub_repo_name": "review_sentiment_analysis_endToEnd_mlops",
    "experiment_name": "Bow vs TfIdf"
}

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"],
             repo_name=CONFIG["dagshub_repo_name"],
             mlflow=True)
mlflow.set_experiment(CONFIG['experiment_name'])


# data preprocessing
# normalize emoji's and enoticons
def normalize_emojis_and_emoticons(text):
    text = str(text)

    # Convert emojis to descriptive text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Normalize common emoticons
    emoticons = {
        r":\)": " smiley face ",
        r":-\)": " smiley face ",
        r":\(": " sad face ",
        r":-\(": " sad face ",
        r";\)": " wink face ",
        r":D": " laughing face ",
        r"<3": " heart "
    }

    for pattern, replacement in emoticons.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# expand contraction words
def expand_contraction(text):
      return contractions.fix(str(text))

# handling missing and invalid reviews 
def handleMissingInvalid(df):
      df = df.dropna(subset=["review"])
      df["review"] = df["review"].astype(str)
      df = df[df["review"].str.strip().ne("")]
      return df

# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    stop_words.difference_update({"no" , "not" , "nor" , "never" })
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    return url_pattern.sub("", text)


def normalize_text(df):
    """Normalize review text for sentiment analysis."""
    try:
        df = handleMissingInvalid(df).copy()

        # Remove URLs before cleaning punctuation
        df["review"] = df["review"].apply(removing_urls)

        # Normalize sentiment-bearing symbols and contractions
        df["review"] = df["review"].apply(
            normalize_emojis_and_emoticons
        )
        df["review"] = df["review"].apply(expand_contraction)

        # Standardize text
        df["review"] = df["review"].apply(lower_case)
        df["review"] = df["review"].apply(removing_punctuations)
        df["review"] = df["review"].apply(removing_numbers)

        # Remove selected stop words and lemmatize
        df["review"] = df["review"].apply(remove_stop_words)
        df["review"] = df["review"].apply(lemmatization)

        return df

    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise
  
def load_data(file_path):

      try:
            df = pd.read_csv(file_path)
            df['sentiment'] = df['sentiment'].map({'positive':1 , 'negative':0})
            df = normalize_text(df)
            return df
      except Exception as e:
            print(f"error occured at loading data : {e}")
            raise 


VECTORIZERS = {
    "BoW": CountVectorizer(
        ngram_range=(1, 2),
        min_df=2
    ),
    "TF-IDF": TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2
    )
}


ALGORITHMS = {
    "LogisticRegression": LogisticRegression(
        random_state=42
    ),
    "MultinomialNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "LinearSVC": LinearSVC(),
    "SGDClassifier": SGDClassifier(
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        n_jobs=-1
    ),
    "RandomForest": RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    )
}

def train_and_evaluate(df):
      with mlflow.start_run() as parent_run:
            for algo_name , algorithm in ALGORITHMS.items():
                  for vec_name , vectorizer in VECTORIZERS.items():
                        with mlflow.start_run(run_name=f"{algo_name} with {vec_name}" , nested=True ) as child_run:
                              try:
                                    X = df['review']
                                    y = df['sentiment']
                                    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=CONFIG['test_size'] , random_state=42)
                                    X_train = vectorizer.fit_transform(X_train)
                                    X_test = vectorizer.transform(X_test)

                                    mlflow.log_params({
                                          "vectorizer" : vec_name,
                                          "algorithm" : algo_name,
                                          "test_size" : CONFIG['test_size']
                                    })

                                    model = algorithm
                                    model.fit(X_train , y_train)

                                    log_model_params(model)

                                    y_pred = model.predict(X_test)

                                    metrics = {
                                          "accuracy": accuracy_score(y_test , y_pred ),
                                          "precision": precision_score(y_test , y_pred ),
                                          "recall": recall_score(y_test , y_pred ),
                                          "f1_score": f1_score(y_test , y_pred )
                                    }

                                    mlflow.log_metrics(metrics)

                                    input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                                    
                                    trusted_types = [
                                        "sklearn.tree._tree.Tree",
                                    ]

                                    if algo_name == "XGBoost":
                                        trusted_types = [
                                            "xgboost.core.Booster",
                                            "xgboost.sklearn.XGBClassifier",
                                        ]

                                    mlflow.sklearn.log_model(
                                        sk_model=model,
                                        name=f"{algo_name}_model",
                                        input_example=input_example,
                                        skops_trusted_types=trusted_types,
                                    )
                                    
                                    print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                                    print(f"Metrics: {metrics}")                            

                              except Exception as e:
                                    print(f"error occured {e}")

def log_model_params(model):
    """Log model hyperparameters to MLflow automatically."""

    params_to_log = model.get_params()

    # Prefix parameters with the algorithm name if needed
    params_to_log = {
        key: value
        for key, value in params_to_log.items()
        if value is not None
    }

    mlflow.log_params(params_to_log)
    
if __name__ == "__main__":
      df = load_data(CONFIG['data_path'])
      train_and_evaluate(df)






      
      
      
