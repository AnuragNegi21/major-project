import os
import sys
from flwr.client import start_numpy_client, NumPyClient
from tensorflow import keras as ks
from utils import load_partition

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IMG_SIZE = 160
DEFAULT_SERVER_ADDRESS = "[::]:8080"

# Define model
model = ks.Sequential([
    ks.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    ks.layers.Dense(128, activation='relu'),  # Reduce neurons
    ks.layers.Dense(4),
    ks.layers.Dense(4)
])
optimizer = ks.optimizers.Adam(learning_rate=0.0001) 
model.compile(
    optimizer=optimizer,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Validate command-line arguments and load data
if len(sys.argv) > 1:
    try:
        partition_number = int(sys.argv[1])
        print(f"Loading partition {partition_number}...")
        X_train, X_val, y_train, y_val = load_partition(partition_number)

        # Limit dataset size to avoid resource issues
        X_train, y_train = X_train[:50], y_train[:50]
        X_val, y_val = X_val[:10], y_val[:10]
    except (ValueError, IndexError):
        print("Invalid partition number. Please provide a valid integer.")
        sys.exit(1)
else:
    print("Usage: python3 client.py PARTITION_NUMBER (e.g., 0, 1, 2, 3)")
    sys.exit(1)


# Define federated client
class FederatedClient(NumPyClient):
    def get_parameters(self, config):
        """Return model weights."""
        return model.get_weights()

    def fit(self, parameters, config):
        """Train model with the provided parameters."""
        model.set_weights(parameters)
        
        try:
            # Train the model
            history = model.fit(
                X_train,
                y_train,
                epochs=config.get("local_epochs", 10),  # Use config for flexibility
                batch_size=config.get("batch_size", 4),  # Reduce batch size further if needed
                validation_data=(X_val, y_val),
                shuffle=True,  # Shuffle data to improve convergence
                verbose=1
            )
            results = {
                "loss": history.history["loss"][-1],
                "accuracy": history.history["accuracy"][-1],
                "val_loss": history.history["val_loss"][-1],
                "val_accuracy": history.history["val_accuracy"][-1],
            }
            print(f"Training results: {results}")
            return model.get_weights(), len(X_train), results
        except Exception as e:
            print(f"Training failed: {e}")
            # Return default weights in case of failure to avoid stopping federated learning
            return model.get_weights(), 0, {}

    def evaluate(self, parameters, config):
        """Evaluate model."""
        model.set_weights(parameters)
        try:
            loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")
            return loss, len(X_val), {"accuracy": accuracy}
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return float('inf'), 0, {"accuracy": 0.0}


# Start federated client
if __name__ == '__main__':
    try:
        start_numpy_client(server_address=DEFAULT_SERVER_ADDRESS, client=FederatedClient())
    except Exception as e:
        print(f"Failed to start the client: {e}")
