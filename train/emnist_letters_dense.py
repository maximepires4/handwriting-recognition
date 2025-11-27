from pathlib import Path

import numpy as np
from dataset import load_emnist

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Dense, Dropout, BatchNormalization
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.serialization import save_model

if __name__ == "__main__":
    print("Classification example: EMNIST Letters Dataset (Dense)")
    seed = 69
    np.random.seed(seed)

    print("Loading data (this may take a while first time)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_emnist(
        split="letters", conv=False
    )

    num_classes = y_train.shape[1]
    print(
        f"Data loaded. Training on {X_train.shape[0]} samples. Number of classes: {num_classes}"
    )

    network = [
        Dense(800, input_size=784),
        BatchNormalization(),
        ReLU(),
        Dropout(0.2),
        Dense(800),
        BatchNormalization(),
        ReLU(),
        Dropout(0.3),
        Dense(num_classes),
    ]

    model = Model(network, CategoricalCrossEntropy(), Adam())

    model.train(
        X_train,
        y_train,
        epochs=15,
        batch_size=128,
        evaluation=(X_val, y_val),
        early_stopping=5,
    )

    print("Evaluating on test set...")
    model.test(X_test, y_test)

    Path("output/").mkdir(parents=True, exist_ok=True)

    save_model(model, "output/dense_emnist_letters.npz")
    print("Model saved to output/dense_emnist_letters.npz")
