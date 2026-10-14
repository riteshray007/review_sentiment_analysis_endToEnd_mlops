import mlflow
import mlflow.sklearn
import re
import html
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
import contractions
import emoji

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import loguniform
import numpy as np
import pandas as pd

print("inside exp file ")
CONFIG = {
    "data_path": "notebooks/IMDB.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/riteshray007/review_sentiment_analysis_endToEnd_mlops.mlflow",
    "dagshub_repo_owner": "riteshray007",
    "dagshub_repo_name": "review_sentiment_analysis_endToEnd_mlops",
    "experiment_name": "SGDClassifier-TFIDF-Tuning"
}

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"],
             repo_name=CONFIG["dagshub_repo_name"],
             mlflow=True)
mlflow.set_experiment(CONFIG['experiment_name'])


# data preprocessing

def validate_and_normalize_labels(df):
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


def clean_review(text):
    """Conservative text normalization for sentiment analysis."""

    # 1. Handle missing values safely
    if text is None:
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
        r"(?<!\w):-?\)": " POSITIVEEMOTION ",
        r"(?<!\w);-?\)": " POSITIVEEMOTION ",
        r"(?<!\w):-?\(": " NEGATIVEEMOTION ",
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



def normalize_text(df):
    df = validate_and_normalize_labels(df)

    df["review"] = df["review"].apply(clean_review)

    # Remove reviews that become empty after cleaning
    df = df[df["review"].str.strip().ne("")].copy()

    return df
 

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("sgd", SGDClassifier(random_state=42))
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
    "sgd__loss": ["hinge", "log_loss", "modified_huber"],
    "sgd__alpha": loguniform(1e-6, 1e-3),
    "sgd__penalty": ["l2", "l1", "elasticnet"],
    "sgd__class_weight": [None, "balanced"],
    "sgd__average": [True, False],
    
    # Additional convergence parameters
    "sgd__tol": [1e-3, 1e-4, 1e-5],
    "sgd__max_iter": [1000, 2000, 3000],
    
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring={
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    },
    refit="f1_macro",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=2,
    return_train_score=True,
    error_score="raise",
)

def mlflow_experiment_tracking(df):
    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=42,
        stratify=y,
    )

    with mlflow.start_run(
        run_name="3rd-SGDC-TFIDF-RSearch_added_params"
    ) as parent_run:

        mlflow.set_tags({
            "project": "review_sentiment_analysis",
            "algorithm": "SGDClassifier",
            "vectorizer": "TF-IDF",
            "search_method": "RandomizedSearchCV",
        })                                                              

        mlflow.log_params({
            "n_iter": 30,
            "cv_folds": 5,
            "test_size": CONFIG["test_size"],
            "random_state": 42,
            "selection_metric": "f1_macro",
        })

        # Run hyperparameter tuning.
        random_search.fit(X_train, y_train)

        print("Best parameters:")
        print(random_search.best_params_)

        print("Best CV macro-F1:", random_search.best_score_)

        # Log every candidate as a nested child run.
        results = random_search.cv_results_

        # Select the top 15 candidates by mean CV macro-F1.
        top_n = 15

        top_indices = np.argsort(
            results["mean_test_f1_macro"]
        )[::-1][:top_n]
        
        print(top_indices)

        for rank, idx in enumerate(top_indices, start=1):
        
            # Retrieve the actual hyperparameters for this candidate.
            params = results["params"][idx]

            with mlflow.start_run(
                run_name=f"Top-{rank}-SGD-TFIDF",
                nested=True,
            ):
                mlflow.set_tags({
                    "ranking_metric": "mean_cv_f1_macro",
                    "rank": str(rank),
                })

                # Log this candidate's hyperparameters.
                mlflow.log_params({
                    key: str(value)
                    for key, value in params.items()
                })

                # Log metrics for the same candidate.
                mlflow.log_metrics({
                    # Cross-validation validation metrics
                    "mean_cv_f1_macro": float(
                        results["mean_test_f1_macro"][idx]
                    ),
                    "std_cv_f1_macro": float(
                        results["std_test_f1_macro"][idx]
                    ),
                    "mean_cv_accuracy": float(
                        results["mean_test_accuracy"][idx]
                    ),
                    "mean_cv_precision_macro": float(
                        results["mean_test_precision_macro"][idx]
                    ),
                    "mean_cv_recall_macro": float(
                        results["mean_test_recall_macro"][idx]
                    ),

                    # Training metrics
                    "mean_train_f1_macro": float(
                        results["mean_train_f1_macro"][idx]
                    ),
                    "mean_train_accuracy": float(
                        results["mean_train_accuracy"][idx]
                    ),
                    "mean_train_precision_macro": float(
                        results["mean_train_precision_macro"][idx]
                    ),
                    "mean_train_recall_macro": float(
                        results["mean_train_recall_macro"][idx]
                    ),
                })

        print(f"Logged the top {len(top_indices)} candidates.")



        # Evaluate the selected model on the held-out test set.
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        test_metrics = {
            "test_accuracy": float(
                accuracy_score(y_test, y_pred)
            ),
            "test_precision_macro": float(
                precision_score(  
                    y_test, y_pred,
                    average="macro", zero_division=0
                )
            ),
            "test_recall_macro": float(
                recall_score(
                    y_test, y_pred,
                    average="macro", zero_division=0
                )
            ),
            "test_f1_macro": float(
                f1_score(
                    y_test, y_pred,
                    average="macro", zero_division=0
                )
            ),
            "best_cv_f1_macro": float(random_search.best_score_),
        }

        # Explicitly log the selected configuration and metrics.
        mlflow.log_params({
            key: str(value)
            for key, value in random_search.best_params_.items()
        })
        mlflow.log_metrics(test_metrics)

        # Log the selected pipeline as an artifact.
        mlflow.sklearn.log_model(
            best_model,
            name="best_sgd_tfidf_model",
            serialization_format="cloudpickle",
        )

        print("Held-out test metrics:", test_metrics)
        print("Parent run ID:", parent_run.info.run_id)



def experimnet_consoling(df):
    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=42,
        stratify=y,
    )

    # # Disable automatic tuning-run creation.
    # mlflow.sklearn.autolog(disable=True)

    # with mlflow.start_run(
    #     run_name="SGDClassifier-TFIDF-RandomizedSearch"
    # ) as parent_run:

    #     mlflow.set_tags({
    #         "project": "review_sentiment_analysis",
    #         "algorithm": "SGDClassifier",
    #         "vectorizer": "TF-IDF",
    #         "search_method": "RandomizedSearchCV",
    #     })

    #     mlflow.log_params({
    #         "n_iter": 30,
    #         "cv_folds": 5,
    #         "test_size": CONFIG["test_size"],
    #         "random_state": 42,
    #         "selection_metric": "f1_macro",
    #     })

        # Run hyperparameter tuning.
    random_search.fit(X_train, y_train)
    print("Best parameters:")
    print(random_search.best_params_)
    print("Best CV macro-F1:", random_search.best_score_)
    # Log every candidate as a nested child run.
    results = random_search.cv_results_
    print(results)
    # for i, params in enumerate(results["params"]):
    #     with mlflow.start_run(
    #         run_name=f"SGD-TFIDF-Candidate-{i + 1}",
    #         nested=True,
    #     ):
            # Convert parameter values to strings for
            # consistent logging of tuples and NumPy values.
            # mlflow.log_params({
            #     key: str(value)
            #     for key, value in params.items()
            # })
            # mlflow.log_metrics({
            #     "mean_cv_f1_macro": float(
            #         results["mean_test_f1_macro"][i]
            #     ),
            #     "std_cv_f1_macro": float(
            #         results["std_test_f1_macro"][i]
            #     ),
            #     "mean_cv_accuracy": float(
            #         results["mean_test_accuracy"][i]
            #     ),
            #     "mean_cv_precision_macro": float(
            #         results["mean_test_precision_macro"][i]
            #     ),
            #     "mean_cv_recall_macro": float(
            #         results["mean_test_recall_macro"][i]
            #     ),
            # })
    # Evaluate the selected model on the held-out test set.
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_metrics = {
        "test_accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "test_precision_macro": float(
            precision_score(
                y_test, y_pred,
                average="macro", zero_division=0
            )
        ),
        "test_recall_macro": float(
            recall_score(
                y_test, y_pred,
                average="macro", zero_division=0
            )
        ),
        "test_f1_macro": float(
            f1_score(
                y_test, y_pred,
                average="macro", zero_division=0
            )
        ),
        "best_cv_f1_macro": float(random_search.best_score_),
    }
    # Explicitly log the selected configuration and metrics.
    # mlflow.log_params({
    #     key: str(value)
    #     for key, value in random_search.best_params_.items()
    # })
    # mlflow.log_metrics(test_metrics)
    # Log the selected pipeline as an artifact.
    # mlflow.sklearn.log_model(
    #     best_model,
    #     name="best_sgd_tfidf_model",
    #     serialization_format="cloudpickle",
    # )
    print("Held-out test metrics:", test_metrics)
    # print("Parent run ID:", parent_run.info.run_id)



# def mlflow_experiment_tracking(df):
#     X=df['review']
#     y=df['sentiment']
#     X_train , X_test , y_train , y_test  = train_test_split(X , y , test_size=CONFIG['test_size'] , random_state=42 , stratify=y )    
    
#     # Enable automatic logging of search results and child runs.
#     # Use cloudpickle to avoid the skops trusted-types issue
#     # encountered earlier in your project.
#     mlflow.sklearn.autolog(
#         log_models=True,
#         max_tuning_runs=None,
#         serialization_format="cloudpickle",
#     )


#     with mlflow.start_run(
#         run_name="SGDClassifier-TFIDF-RandomizedSearch"
#     ) as parent_run:

#         mlflow.set_tag("project", "review_sentiment_analysis")
#         mlflow.set_tag("algorithm", "SGDClassifier")
#         mlflow.set_tag("vectorizer", "TF-IDF")
#         mlflow.set_tag("search_method", "RandomizedSearchCV")

#         mlflow.log_param("n_iter", 30)
#         mlflow.log_param("cv_folds", 5)
#         mlflow.log_param("selection_metric", "f1_macro")

#         # Fit performs cross-validation and creates child runs via autologging
#         random_search.fit(X_train, y_train)

#         print("Best parameters:")
#         print(random_search.best_params_)

#         print("\nBest cross-validation macro-F1:")
#         print(random_search.best_score_)

#         # Evaluate the refitted best pipeline on the held-out test set
#         best_model = random_search.best_estimator_
#         y_pred = best_model.predict(X_test)

#         test_metrics = {
#             "test_accuracy": accuracy_score(y_test, y_pred),
#             "test_precision_macro": precision_score(
#                 y_test, y_pred, average="macro", zero_division=0
#             ),
#             "test_recall_macro": recall_score(
#                 y_test, y_pred, average="macro", zero_division=0
#             ),
#             "test_f1_macro": f1_score(
#                 y_test, y_pred, average="macro", zero_division=0
#             ),
#         }

#         mlflow.log_metric(
#             "best_cv_f1_macro",
#             random_search.best_score_
#         )
#         mlflow.log_metrics(test_metrics)

#         print("\nHeld-out test metrics:")
#         print(test_metrics)

#         print("\nParent run ID:", parent_run.info.run_id)
#         print("Best parameters:", random_search.best_params_)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    df = load_data("notebooks/IMDB.csv")
    df = normalize_text(df)
    # print(df.head())
    # print(type(df))
    mlflow_experiment_tracking(df)