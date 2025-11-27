from pathlib import Path

import numpy as np

from mpneuralnetwork import serialization
from mpneuralnetwork.layers import Dense, Convolutional


class NeuralNetHandler:
    def __init__(self, model_path: Path):
        self.model = serialization.load_model(model_path)

    def predict(self, image):
        img = np.array(image).astype(np.float32) / 255.0
        
        # Auto-detect input shape based on first layer
        first_layer = self.model.layers[0]
        
        if isinstance(first_layer, Convolutional):
            # Keep 3D shape for CNNs (Channel, Height, Width)
            input_data = img.reshape(1, 28, 28)
        else:
            # Default/Flatten for Dense networks
            input_data = img.reshape(784)
            
        return self.model.predict(input_data)
