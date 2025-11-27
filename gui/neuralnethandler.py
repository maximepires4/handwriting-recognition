from pathlib import Path

import numpy as np

from mpneuralnetwork import serialization
from mpneuralnetwork.layers import Convolutional


class NeuralNetHandler:
    def __init__(self, model_path: Path):
        self.model = serialization.load_model(model_path)

    def predict(self, image):
        img = np.array(image).astype(np.float32) / 255.0

        first_layer = self.model.layers[0]

        if isinstance(first_layer, Convolutional):
            input_data = img.reshape(1, 28, 28)
        else:
            input_data = img.reshape(784)

        return self.model.predict(input_data)

    @property
    def output_size(self):
        for layer in reversed(self.model.layers):
            weights = getattr(layer, "weights")
            if weights is not None:
                return weights.shape[1]

        return 10
