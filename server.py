import flwr as fl
from tensorflow import keras as ks
from typing import List, Dict
import numpy as np

from utils import load_testing_data

DEFAULT_SERVER_ADDRESS = "[::]:8080"
IMG_SIZE = 160

# Define a centralized evaluation function for the server
def evaluate_fn(server_round, parameters, config):
    model.set_weights(parameters)
    X_test, y_test = load_testing_data()
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"****** CENTRALIZED ACCURACY (Round {server_round}): {accuracy:.4f} ******")
    return loss, {"accuracy": accuracy}


# Load our model on the server
model = ks.Sequential([
    ks.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    ks.layers.Dense(128, activation='relu'),
    ks.layers.Dense(4),
    ks.layers.Dense(4)
])
optimizer = ks.optimizers.Adam(learning_rate=0.0001) 
model.compile(
    optimizer=optimizer,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Create strategy and pass the centralized evaluation and aggregation functions
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,  # Sample 50% of available clients per round
    min_fit_clients=1,
    min_available_clients=2,)


if __name__ == '__main__':
    fl.server.start_server(
        server_address=DEFAULT_SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
