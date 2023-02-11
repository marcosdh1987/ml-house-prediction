# This is a script to serve a model using mlflow and serve for Flask usage

# load the dataset
import pickle
import warnings

import mlflow

warnings.filterwarnings("ignore")
mlflow.set_registry_uri("sqlite:///mlruns.db")


def save_model(rf):
    # save the model to disk
    filename = "../models/model.pkl"
    pickle.dump(rf, open(filename, "wb"))


if __name__ == "__main__":

    try:
        logged_model = "models:/HousePriceModel/production"
        rf = mlflow.sklearn.load_model(logged_model)
        save_model(rf)
        print("Model saved")
    except Exception as e:
        print("Error while saving the model")
        print(e)
