from pathlib import Path

import numpy as np

from mpneuralnetwork import model


class NeuralNetHandler:
    def __init__(self, path: Path = Path("output/model.npz")):
        self.model = model.Model.load(path)

    def predict(self, image):
        img = np.array(image).reshape(784)
        return self.model.predict(img)
