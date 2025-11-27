import argparse
from pathlib import Path
from gui.paint import Paint
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwriting Recognition GUI")

    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default="output/dense_mnist.npz",
        help="Path to the trained model file (.npz)",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train a model first using scripts in the train/ directory.")
        sys.exit(1)

    print(f"Starting GUI with model: {model_path}")
    Paint(model_path)