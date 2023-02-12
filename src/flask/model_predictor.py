# load the dataset
import json
import pickle
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
# load the model from pickle
rf = pickle.load(open("src/models/model.pkl", "rb"))


def preprocess(payload):
    # preprocess the data
    print("Preprocessing the data")
    # load the preprocessor
    preprocessor = pickle.load(open("src/models/preprocessor.pkl", "rb"))
    # load the column names
    with open("src/models/columns.json", "r") as f:
        columns = json.load(f)
    # create a dataframe from the payload and columns
    df = pd.DataFrame([payload.split(",")], columns=columns)
    # transform the data
    data = preprocessor.transform(df)
    return data


def make_pred(data_processed):

    print("Predicting the data")
    # make predictions
    predictions = rf.predict(data_processed)
    return predictions
