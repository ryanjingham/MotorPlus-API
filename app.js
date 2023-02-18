const express = require("express");
const app = express();
const port = process.env.PORT || 3000;
const config = require("./config.json");
const API_KEY = config.API_KEY;
const keras = require("keras");

app.post("/predict_keras", (req, res) => {
  // Get the API key from the request header
  const apiKey = req.header("X-API-KEY");

  // Check if the API key is valid
  if (apiKey !== API_KEY) {
    return res.status(401).send({ error: "Unauthorized" });
  }

  // Load the saved Keras model
  const model = keras.models.load_model(
    "./MPGClassifier/models/model_keras.h5"
  );

  // Get the input data from the request body
  const inputData = req.body.input_data;

  // Use the loaded model to make a prediction
  const predictionResult = model.predict(inputData);

  // Return the prediction result as the response
  res.send({ prediction: predictionResult });
});

app.listen(port, () => {
  console.log("API server started on port ${port}");
});
