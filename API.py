from flask import Flask, request, abort
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os
from apiauth import generate_api_key, load_user_api_keys, save_user_api_keys

app = Flask(__name__)
model_metadata_file = "model_metadata.json"
user_api_keys_file = "user_api_keys.json"

latest_model_path = "models_prod/model_keras_latest.h5"
model = keras.models.load_model(latest_model_path)

if not os.path.exists(model_metadata_file):
    with open(model_metadata_file, "w") as metadata_file:
        json.dump(
            {"latest_model": {"name": "Latest Model", "path": latest_model_path}},
            metadata_file,
        )

user_api_keys = {}


@app.route("/upload_model", methods=["POST"])
def upload_model():
    api_key = request.headers.get("X-API-Key")
    if api_key not in user_api_keys.values():
        abort(401, "Unauthorised")
    # Retrieve the uploaded model file and new model name from the request
    uploaded_file = request.files["model_file"]
    new_model_name = request.form["model_name"]

    # Define the path where the new model will be saved
    new_model_path = f"models_prod/{new_model_name}.h5"

    # Save the uploaded model file to the specified path
    uploaded_file.save(new_model_path)

    # Load the existing model metadata
    with open(model_metadata_file, "r") as metadata_file:
        model_metadata = json.load(metadata_file)

    # Add the metadata of the new model to the existing metadata dictionary
    model_metadata[new_model_name] = {"name": new_model_name, "path": new_model_path}

    # Update the model metadata file with the new model metadata
    with open(model_metadata_file, "w") as metadata_file:
        json.dump(model_metadata, metadata_file)

    # Reload the global model with the newly uploaded model
    global model
    model = keras.models.load_model(new_model_path)

    return "Model uploaded and updated successfully"


@app.route("/predict_keras", methods=["POST"])
def predict_keras():
    api_key = request.headers.get("X-API-Key")
    if api_key not in user_api_keys.values():
        abort(401, "Unauthorised")
    # Retrieve the vehicle data from the request
    input_data = request.json.get("input_data")

    # Convert the data to a pandas dataframe for processing
    df = pd.DataFrame.from_dict(input_data, orient="index").transpose()
    print(df)

    # One-hot encoding on the "origin" (nation) column
    df = pd.get_dummies(df, columns=["origin"])

    # Scale the input data
    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    # Make predictions
    prediction_result = model.predict(df)
    print(prediction_result)

    # Convert the prediction results to a list and return as the response
    return {"prediction": prediction_result.tolist()}


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
