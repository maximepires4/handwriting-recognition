from pathlib import Path

import numpy as np
from dataset import load_mnist

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Dense, Dropout, BatchNormalization
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.serialization import save_model

if __name__ == "__main__":
    print("Classification example: MNIST Dataset")
    seed = 69
    np.random.seed(seed)

    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()

    print(f"Data loaded. Training on {X_train.shape[0]} samples.")

    network = [
        Dense(800, input_size=784),
        BatchNormalization(),
        ReLU(),
        Dropout(0.2),
        Dense(800),
        BatchNormalization(),
        ReLU(),
        Dropout(0.3),
        Dense(10),
    ]

    model = Model(network, CategoricalCrossEntropy(), Adam())

    model.train(
        X_train,
        y_train,
        epochs=20,
        batch_size=128,
        evaluation=(X_val, y_val),
        early_stopping=5,
    )

    print("Evaluating on test set...")
    model.test(X_test, y_test)

    Path("output/").mkdir(parents=True, exist_ok=True)

    save_model(model, "output/dense_mnist.npz")
    print("Model saved to output/dense_mnist.npz")
