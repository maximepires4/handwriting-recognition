import gzip
import shutil
import urllib.request
import zipfile
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


def load_emnist(split="balanced", conv=False):
    """
    Downloads and loads the EMNIST dataset.

    Args:
        split (str): The split to load ('balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist').
                     Defaults to 'balanced'.
        conv (bool): If True, returns data in (N, 1, 28, 28) format.
                     If False, returns (N, 784).

    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    valid_splits = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}")

    base_path = Path("data/emnist")
    base_path.mkdir(parents=True, exist_ok=True)

    # Files we need for the requested split
    files_map = {
        "x_train": f"emnist-{split}-train-images-idx3-ubyte.gz",
        "y_train": f"emnist-{split}-train-labels-idx1-ubyte.gz",
        "x_test": f"emnist-{split}-test-images-idx3-ubyte.gz",
        "y_test": f"emnist-{split}-test-labels-idx1-ubyte.gz",
    }

    # Check if we need to download/extract
    missing_files = [f for f in files_map.values() if not (base_path / f).exists()]

    if missing_files:
        zip_name = "emnist-gzip.zip"
        zip_path = base_path / zip_name
        zip_url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

        if not zip_path.exists():
            print(
                f"Downloading EMNIST ({split}). This might take a while (approx 530MB)..."
            )
            try:
                get_file("https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip", zip_path)
            except Exception as e:
                print(
                    "Download failed. Please check your internet connection or the URL."
                )
                if zip_path.exists():
                    zip_path.unlink()
                raise e

        print(f"Extracting files for split '{split}'...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file_name in files_map.values():
                # files are inside a 'gzip/' folder in the zip
                member_name = f"gzip/{file_name}"
                try:
                    with (
                        zip_ref.open(member_name) as source,
                        open(base_path / file_name, "wb") as target,
                    ):
                        shutil.copyfileobj(source, target)
                except KeyError:
                    print(f"Error: Could not find {member_name} in zip.")
                    raise

    # Load data
    data = {}
    offsets = {"x_train": 16, "y_train": 8, "x_test": 16, "y_test": 8}

    for key, filename in files_map.items():
        p = base_path / filename
        with gzip.open(p, "rb") as f_in:
            data[key] = np.frombuffer(f_in.read(), dtype=np.uint8, offset=offsets[key])

    def process_x(x):
        # EMNIST images are rotated 90 degrees clockwise and flipped.
        # 1. Reshape to (N, 28, 28)
        x = x.reshape(-1, 28, 28)
        # 2. Transpose (swap height and width) to fix rotation
        x = x.transpose(0, 2, 1)

        if conv:
            x = x.reshape(-1, 1, 28, 28)
        else:
            x = x.reshape(-1, 784)

        return x.astype(np.float32) / 255.0

    X_train = process_x(data["x_train"])
    X_test = process_x(data["x_test"])

    # Labels
    y_train_raw = data["y_train"]
    y_test_raw = data["y_test"]

    num_classes = int(max(y_train_raw.max(), y_test_raw.max()) + 1)

    y_train = np.eye(num_classes)[y_train_raw]
    y_test = np.eye(num_classes)[y_test_raw]

    # Create validation split (10% of training data)
    split_idx = int(len(X_train) * 0.9)

    return (
        (X_train[:split_idx], y_train[:split_idx]),
        (X_train[split_idx:], y_train[split_idx:]),
        (X_test, y_test),
    )