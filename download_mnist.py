import sys
import urllib.request
from pathlib import Path


def download_mnist():
    data_dir = Path("data")

    data_dir.mkdir(exist_ok=True)

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for filename in files:
        filepath = data_dir / filename
        url = base_url + filename

        if filepath.exists():
            print(f"File already exists: {filepath}")
            continue

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    download_mnist()
