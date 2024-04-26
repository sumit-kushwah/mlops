from sklearn.preprocessing import StandardScaler
import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

parser = argparse.ArgumentParser(description="Model Training")

parser.add_argument("--x-train-path", type=str, help="X train folder", required=True)
parser.add_argument("--y-train-path", type=str, help="y train folder", required=True)
parser.add_argument("--model-out-dir", type=str, help="Model Directory", required=True)
parser.add_argument(
    "--mlflow_uri",
    type=str,
    help="MLFlow Tracking URI",
    default="http://localhost:8080",
)
parser.add_argument(
    "--mlflow_exp",
    type=str,
    help="MLFlow Experiment Name",
)

args = parser.parse_args()

x_train_path = os.path.join(args.x_train_path, "X_train.csv")
y_train_path = os.path.join(args.y_train_path, "y_train.csv")
model_path = args.model_out_dir
mlflow_tracking_uri = args.mlflow_uri
experiment_name = args.mlflow_exp

if os.path.exists(x_train_path):
    print(f"x train path found: {x_train_path}")
    input_file = os.path.abspath(x_train_path)
else:
    print(f"Input file not found: {x_train_path}")
    sys.exit(1)

if os.path.exists(y_train_path):
    print(f"x train path found: {y_train_path}")
    input_file = os.path.abspath(y_train_path)
else:
    print(f"Input file not found: {y_train_path}")
    sys.exit(1)


if os.path.exists(model_path):
    print(f"Output file found: {model_path}")
else:
    print(f"Output path not found: {model_path}")
    print("Creating output path")
    os.makedirs(model_path, exist_ok=True)

model_path = os.path.abspath(model_path)

X_train = pd.read_csv(x_train_path)
y_train = pd.read_csv(y_train_path)


# MLFlow tracking setup
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(experiment_name)


# Model Training
mlflow.autolog()
# Importing necessary libraries
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

# Define the neural network architecture
model = Sequential(
    [
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(128, activation="relu", kernel_initializer="random_uniform"),
        Dense(64, activation="relu", kernel_initializer="random_uniform"),
        Dense(64, activation="relu", kernel_initializer="random_uniform"),
        Dense(32, activation="relu", kernel_initializer="random_uniform"),
        Dense(16, activation="relu", kernel_initializer="random_uniform"),
        Dense(1),  # Output layer with 1 neuron for regression
    ]
)

with mlflow.start_run(run_name="Model Training(NN)") as run:
    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Display the model summary
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32)

    # saving the nn model
    model.save(os.path.join(model_path, "nn_model.h5"))
