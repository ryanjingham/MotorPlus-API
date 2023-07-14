from flask import Flask, request, abort
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import os
from apiauth import generate_api_key, load_user_api_keys, save_user_api_keys

app = Flask(__name__)
model_metadata_file = "model_metadata.json"
user_api_keys_file = "user_api_keys.json"

latest_model_path = "models_prod/neural_network_model.pkl"

if not os.path.exists(model_metadata_file):
    with open(model_metadata_file, "w") as metadata_file:
        json.dump(
            {"latest_model": {"name": "Latest Model", "path": latest_model_path}},
            metadata_file,
        )

user_api_keys = {}


scaler = MinMaxScaler()
training_data_file = "Datasets/auto-mpg.csv"
data = pd.read_csv(training_data_file)
data = data.replace("?", np.nan)
data = data.dropna()
X_train = data.drop("mpg", axis=1).values
scaler.fit(X_train)


@app.route("/predict_mpg", methods=["POST"])
def predict_mpg():
    input_data = request.json.get("input_data")

    cylinders = input_data.get("cylinders")
    displacement = input_data.get("displacement")
    horsepower = input_data.get("horsepower")
    weight = input_data.get("weight")
    acceleration = input_data.get("acceleration")
    year = input_data.get("year")

    input_features = np.array(
        [[cylinders, displacement, horsepower, weight, acceleration, year]]
    )
    input_features = input_features.astype(float)

    loaded_model = None
    with open(latest_model_path, "rb") as file:
        loaded_model = pickle.load(file)

    prediction = loaded_model.predict(input_features)
    prediction = scaler.inverse_transform(
        prediction.reshape(-1, 1)
    )  # Inverse transform the array of predictions

    prediction_val = prediction[0][0]
    return {"prediction": prediction_val}


@app.route("/health", methods=["GET"])
def health_check():
    api_key = request.headers.get("X-API-Key")
    if api_key not in user_api_keys.values():
        abort(401, "Unauthorised")
    return "OK"


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username")
    api_key = generate_api_key()
    user_api_keys[username] = api_key
    save_user_api_keys(user_api_keys)

    return {"api_key": api_key}


if __name__ == "__main__":
    keys = load_user_api_keys()
    new_key = generate_api_key()
    print(new_key)
    username = "admin"
    keys[username] = new_key
    save_user_api_keys(keys)
    app.run(debug=True, host="0.0.0.0", port=5000)
