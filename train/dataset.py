import gzip
import urllib.request
from pathlib import Path

import numpy as np


def get_file(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)


def load_mnist(conv=False):
    base_path = Path("data/mnist")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "x_train": ("train-images-idx3-ubyte.gz", 16),
        "y_train": ("train-labels-idx1-ubyte.gz", 8),
        "x_test": ("t10k-images-idx3-ubyte.gz", 16),
        "y_test": ("t10k-labels-idx1-ubyte.gz", 8),
    }

    data = {}
    for key, (filename, offset) in files.items():
        p = base_path / filename
        get_file(base_url + filename, p)
        with gzip.open(p, "rb") as f_in:
            data[key] = np.frombuffer(f_in.read(), dtype=np.uint8, offset=offset)

    if conv:
        X_train = data["x_train"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        X_test = data["x_test"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    else:
        X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
        X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0

    y_train = np.eye(10)[data["y_train"]]
    y_test = np.eye(10)[data["y_test"]]

    return (
        (X_train[:50000], y_train[:50000]),
        (X_train[50000:], y_train[50000:]),
        (X_test, y_test),
    )
