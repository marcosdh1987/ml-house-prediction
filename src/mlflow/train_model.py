import logging
import pickle
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def load_train_data():
    """Load data from parquet files."""
    X_train_tf = pd.read_parquet("../data/processed/X_train_tf.parquet")
    y_train = pd.read_parquet("../data/processed/y_train.parquet")
    return X_train_tf, y_train


def load_test_data():
    """Load data from parquet files."""
    X_test_tf = pd.read_parquet("../data/processed/X_test_tf.parquet")
    y_test = pd.read_parquet("../data/processed/y_test.parquet")
    return X_test_tf, y_test


if __name__ == "__main__":
    # Use sqlite:///mlruns.db as the local store for tracking and registery
    warnings.filterwarnings("ignore")
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    # load dataset
    print("Loading Data...")
    X_train_tf, y_train = load_train_data()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n_estimators", type=int, default=10)

    # args = parser.parse_args()
    # n_estimators = args.n_estimators

    # params = {
    #     "n_estimators": n_estimators,
    # }
    # load best params from pkl file
    # load the best parameters
    best_params = pickle.load(open("../models/best_params.pkl", "rb"))

    print("Starting training...")
    with mlflow.start_run():
        rf = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_features=best_params["max_features"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            bootstrap=best_params["bootstrap"],
        )
        rf.fit(X_train_tf, y_train)
        print("Model trained")
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(rf, "model")

    # Evaluate the model
    print("Evaluating the model")
    X_test_tf, y_test = load_test_data()
    y_pred = rf.predict(X_test_tf)

    # Compute and log metrics for test data
    test_rmae = np.sqrt(mean_absolute_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    print("Test  RMAE (Root Mean Squared Error): %s" % test_rmae)
    print("Test  MAE (Mean Absolute Error): %s" % test_mae)
    mlflow.log_metric("test_rmae", test_rmae)
    mlflow.log_metric("test_mae", test_mae)
    # Compute and log metrics for train data
    y_pred = rf.predict(X_train_tf)
    train_rmae = np.sqrt(mean_absolute_error(y_train, y_pred))
    train_mae = mean_absolute_error(y_train, y_pred)
    print("Train RMAE (Root Mean Squared Error): %s" % train_rmae)
    print("Train MAE (Mean Absolute Error): %s" % train_mae)
    mlflow.log_metric("train_rmae", train_rmae)
    mlflow.log_metric("train_mae", train_mae)

    # mlflow.set_tracking_uri("sqlite:///mlruns.db")
    # change the tracking uri to the one you want to use
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(rf, "model", registered_model_name="HousePriceModel")
        print("Model registered")
    else:
        mlflow.sklearn.log_model(rf, "model")
        print("Model logged")
