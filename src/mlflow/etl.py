# Import the necessary libraries
import argparse
import json
import os
import pickle
import re
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from utils.constants import (COLUMNS_FILENAME, PREPROCESSOR_FILENAME,
                             RAW_FILENAME, X_TEST_FILENAME, X_TRAIN_FILENAME,
                             Y_TEST_FILENAME, Y_TRAIN_FILENAME)

warnings.filterwarnings("ignore")

# defining the default path to the dataset
raw_filename = RAW_FILENAME
x_train_filename = X_TRAIN_FILENAME
y_train_filename = Y_TRAIN_FILENAME
x_test_filename = X_TEST_FILENAME
y_test_filename = Y_TEST_FILENAME
preprocessor_filename = PREPROCESSOR_FILENAME
columns_filename = COLUMNS_FILENAME
# get the execution path
execution_path = os.getcwd()
# join the execution path with the path to the dataset
key = os.path.join(execution_path, raw_filename)


def save(X_train_tf, X_test_tf, y_train, y_test):
    X_train_tf.to_parquet(os.path.join(execution_path, x_train_filename))
    X_test_tf.to_parquet(os.path.join(execution_path, x_test_filename))
    # change series to dataframe
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    y_train.to_parquet(os.path.join(execution_path, y_train_filename))
    y_test.to_parquet(os.path.join(execution_path, y_test_filename))


def save_preprocessor(preprocessor, X_train):
    pickle.dump(
        preprocessor, open(os.path.join(execution_path, preprocessor_filename), "wb")
    )
    # save column names in json file
    with open(os.path.join(execution_path, columns_filename), "w") as f:
        json.dump(X_train.columns.tolist(), f)


def etl(filepath):

    df = pd.read_csv(filepath)
    # define target column, and retrieve list of attributes from the dataframe
    id_cols = []  # If the dataset has an ID column, it should be included here
    target_col = "sale_price"
    categ_cols = [
        "overall_quality",
        "overall_condition",
        "liv_lot_ratio",
    ]  # If the dataset has known categorical columns, they should be included here
    other_cols = (
        []
    )  # If the dataset has other columns that should not be included in the model, they should be included here
    feature_cols = [
        x for x in df.columns.tolist() if x not in id_cols + other_cols + [target_col]
    ]
    numeric_features = [
        x
        for x in feature_cols
        if df[x].dtype != "object"
        and not re.match("(^has_)", x)
        and x not in categ_cols
    ]
    categorical_features = [x for x in feature_cols if x not in numeric_features]
    print("Imputing missing values")
    # Imputing missing values for numeric features
    imputer = SimpleImputer(strategy="median")
    df[numeric_features] = imputer.fit_transform(df[numeric_features])
    # X and y definition
    X = df[feature_cols]
    y = df[target_col]
    print("Splitting the dataset into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    # define enconders
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_encoder", ordinal_encoder, categorical_features),
            ("scaler", scaler, numeric_features),
        ],
        remainder="passthrough",
    )
    print("Preprocessing the data")
    # fit the preprocessor to the training data
    X_train_tf = preprocessor.fit_transform(X_train)
    # add the transformed columns to the dataframe
    X_train_tf = pd.DataFrame(X_train_tf, columns=X_train.columns.tolist())
    X_test_tf = preprocessor.transform(X_test)
    # add the transformed columns to the dataframe
    X_test_tf = pd.DataFrame(X_test_tf, columns=X_test.columns.tolist())
    print("Saving the preprocessed data")
    save(X_train_tf, X_test_tf, y_train, y_test)
    print("Saving the preprocessor")
    save_preprocessor(preprocessor, X_train)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Add the arguments to the parser to be used in the etl function if not input by the user use the default value
    parser.add_argument("--filepath", type=str, help="filepath", default=key)
    args = parser.parse_args()
    print("ETL process started")
    print("Reading the dataset from: ", args.filepath)
    etl(key)
    print("ETL process finished")
