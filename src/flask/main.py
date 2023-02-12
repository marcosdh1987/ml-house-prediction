from flask import Flask, jsonify
from src.flask.model_evaluation import get_metrics
from src.flask.model_predictor import make_pred, preprocess

app = Flask(__name__)
app.config["SECRET_KEY"] = "this is a secret key random"
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/model_performance/", methods=["GET"])
def model_performance():
    response = get_metrics()
    return response


@app.route("/predict/", methods=["GET"])
def predict():
    response = {}
    response["predictions"] = 162709.5
    return response


@app.route("/predict/<string:payload>/", methods=["GET"])
def make_prediction(payload):
    # check if payload is passed and contains 17 values
    if payload and len(payload.split(",")) == 17:
        data_processed = preprocess(payload)
        predictions = make_pred(data_processed)
        return jsonify({"predictions": predictions[0]})
    else:
        return jsonify({"error": "Please provide a payload with 17 values"})


if __name__ == "__main__":
    app.run()