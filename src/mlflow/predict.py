# load the dataset
import argparse
import json
import pickle
import warnings

import pandas as pd

import mlflow

warnings.filterwarnings("ignore")
mlflow.set_registry_uri("sqlite:///mlruns.db")
logged_model = "models:/HousePriceModel/production"
rf = mlflow.sklearn.load_model(logged_model)


def preprocess(payload):
    # preprocess the data
    print("Preprocessing the data")
    # load the preprocessor
    preprocessor = pickle.load(open("../models/preprocessor.pkl", "rb"))
    # transform the data
    data = preprocessor.transform(payload)
    print(data)
    return data


def predict(data_processed):

    print("Predicting the data")
    # make predictions
    predictions = rf.predict(data_processed)
    return predictions


if __name__ == "__main__":
    # get arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, help="payload")

    # check if payload is passed
    if parser.parse_args().payload:
        payload = pd.read_json(parser.parse_args().payload)
        data_processed = preprocess(payload)
    else:
        # load example data from json file
        with open("../data/raw/example.json") as f:
            data = json.load(f)
        # create payload with values from json file
        payload = pd.DataFrame(data, index=[0])
        data_processed = preprocess(payload)

    try:
        # make predictions
        predictions = predict(data_processed)
        # show the price prediction
        print("The predicted price is: ", predictions[0])

    except Exception as ex:
        raise (ex)
