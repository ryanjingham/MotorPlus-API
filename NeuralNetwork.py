import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


class NeuralNetwork:
    def __init__(
        self, input_shape, hidden_layer_size, output_shape, learning_rate=0.05
    ):
        self.input_shape = input_shape
        self.hidden_layer_size = hidden_layer_size
        self.output_shape = output_shape
        self.learning_rate = learning_rate

        # Initialise the weights and biases for each layer with random values drawn from a normal distribution
        # self.weights = [
        #     np.random.normal(0, 1, (self.input_shape, self.hidden_layer_size)),
        #     np.random.normal(0, 1, (self.hidden_layer_size, self.output_shape)),
        # ]
        # self.biases = [
        #     np.zeros((1, self.hidden_layer_size)),
        #     np.zeros((1, self.output_shape)),
        # ]

        self.weights = [
            np.random.normal(0, 1, size=(self.input_shape, self.hidden_layer_size)),
            np.random.normal(0, 1, size=(self.hidden_layer_size, self.output_shape)),
        ]
        self.biases = [
            np.zeros((1, self.hidden_layer_size)),
            np.zeros((1, self.output_shape)),
        ]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs):
        errors = []
        for epoch in range(epochs):
            activations = [X]
            zs = []
            for i in range(len(self.weights)):
                # Forward pass
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                a = self.sigmoid(z)
                activations.append(a)
                zs.append(z)

            # Backpropagation
            error = activations[-1] - y
            errors.append(np.mean(np.abs(error)))

            delta = error * self.sigmoid_derivative(activations[-1])
            for i in range(len(self.weights) - 1, -1, -1):
                # Update weights and biases using gradient descent
                self.weights[i] -= self.learning_rate * np.dot(activations[i].T, delta)
                self.biases[i] -= self.learning_rate * np.sum(
                    delta, axis=0, keepdims=True
                )
                if i != 0:
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(
                        activations[i]
                    )

        return errors

    def visualise_errors(self, errors, mae, mse):
        epochs = range(1, len(errors) + 1)
        plt.figure(figsize=(10, 5))

        # Plot Mean Absolute Error (MAE)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, errors, "b.-")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title(f"Mean Absolute Error (MAE): {mae:.4f}")

        # Plot Mean Squared Error (MSE)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [e**2 for e in errors], "r.-")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title(f"Mean Squared Error (MSE): {mse:.4f}")

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            # Forward pass to make predictions
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
        return activations[-1]


# if __name__ == "__main__":
#     # Load the dataset and data preprocessing
#     data = pd.read_csv("Datasets/auto-mpg.csv")
#     scaler = MinMaxScaler()
#     data = data.replace("?", np.nan)
#     data = data.dropna()

#     # Prepare the input features (X) and target variable (y)
#     X = data.drop("mpg", axis=1).values
#     y = data["mpg"].values.reshape(-1, 1)

#     # Scale the input features and target variable
#     X = scaler.fit_transform(X)
#     y = scaler.fit_transform(y)

#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # training parameters
#     input_shape = X_train.shape[1]
#     output_shape = 1
#     epochs = 500

#     # Variables to track the best MAE and corresponding parameters
#     best_mae = np.inf
#     best_params = {}
#     errors_list = []
#     mae_list = []
#     mse_list = []

#     # Loop through different combinations of learning rates and hidden layer sizes
#     for learning_rate in [0.01, 0.05, 0.1]:
#         for hidden_layer_size in [5, 10, 15]:
#             # Create a neural network instance
#             neural_network = NeuralNetwork(
#                 input_shape, hidden_layer_size, output_shape, learning_rate
#             )

#             # Train the neural network and get the errors
#             errors = neural_network.train(X_train, y_train, epochs)
#             errors_list.append(errors)

#             # Make predictions on the test set
#             y_pred = neural_network.predict(X_test)

#             # Calculate the Mean Absolute Error (MAE) and Mean Squared Error (MSE)
#             mae = mean_absolute_error(
#                 scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred)
#             )
#             mse = mean_squared_error(
#                 scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred)
#             )

#             mae_list.append(mae)
#             mse_list.append(mse)

#             # Print the metrics and parameters for each combination
#             print(
#                 f"MAE: {mae} | MSE: {mse} | Params: learning_rate={learning_rate}, hidden_layer_size={hidden_layer_size}"
#             )

#             # Check if the current combination yields the best MAE
#             if mae < best_mae:
#                 best_mae = mae
#                 best_params = {
#                     "learning_rate": learning_rate,
#                     "hidden_layer_size": hidden_layer_size,
#                 }

#     # Plot MAE and MSE for all combinations
#     for i in range(len(errors_list)):
#         neural_network.visualise_errors(errors_list[i], mae_list[i], mse_list[i])

#     # Save the best model
#     best_model = neural_network
#     model_path = "models_qa/neural_network_model.pkl"
#     with open(model_path, "wb") as file:
#         pickle.dump(best_model, file)

#     # Print the best MAE and corresponding parameters
#     print(f"Best MAE: {best_mae} | Best Params: {best_params}")
