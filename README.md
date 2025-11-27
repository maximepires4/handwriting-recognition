# Real-Time Handwriting Recognition

A GUI application that recognizes handwritten digits in real-time.

**This project serves as a proof-of-concept for [MPNeuralNetwork](https://github.com/maximepires4/mp-neural-network/), my custom Deep Learning library built from scratch.**

<center><img src="images/example.gif" width="75%" height="75%"></center>

## Overview

The goal of this project is to demonstrate that a neural network built without frameworks (like TensorFlow or PyTorch) can perform efficiently in a production-like environment.

The application features a drawing canvas where the user inputs a digit.
The app processes the image on the fly and feeds it to a neural network trained on the MNIST dataset.

## Model & Architecture

The "brain" of this application is a Neural Network built with `mp-neural-network`. Three architectures are available in the `train/` directory:

### 1. Super CNN (High Performance) - `train/super_cnn.py`

A deeper Convolutional Neural Network designed for higher accuracy.

- **Architecture:** `Conv2D(32) -> BN -> ReLU -> MaxPool -> Conv2D(64) -> BN -> ReLU -> MaxPool -> Flatten -> Dense(128) -> BN -> ReLU -> Dropout(0.5) -> Dense(10)`
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam

### 2. CNN (Standard) - `train/cnn_mnist.py`

A standard Convolutional Neural Network.

- **Architecture:** `Conv2D(32) -> ReLU -> MaxPool -> Flatten -> Dense(128) -> BN -> ReLU -> Dense(10)`

### 3. Dense (Feed-Forward) - `train/mnist.py`

A classic Multi-Layer Perceptron.

- **Architecture:** `Dense(800) -> BN -> ReLU -> Dropout(0.2) -> Dense(800) -> BN -> ReLU -> Dropout(0.3) -> Dense(10)`

## Image Processing Pipeline

To ensure the model works on user drawings, the input must match the MNIST dataset format strictly.
The `ImageHandler` class performs the following steps (identical to MNIST preprocessing):

1. **Bounding Box Crop:** Isolate the digit from the empty canvas space.
2. **Resize:** Scale the digit down to fit in a 20x20 pixel box while preserving the aspect ratio.
3. **Center of Mass:** Compute the center of mass of the pixels.
4. **Padding:** Place the 20x20 image onto a 28x28 black canvas, positioning it based on the center of mass.

This pipeline makes the recognition robust to drawing size and position.

## Installation

### 1. Prerequisites

Install the dependencies:

```bash
pip install -r requirements.txt
```

*Note for Linux users: You might need to install Tkinter manually:* `sudo apt install python3-tk`

### 2. Train a model

You can train any of the available models by executing a script under `train/`

The application is now run via `run_gui.py`.

## Usage

### Basic Syntax

```bash
python3 run_gui.py [-m <path_to_model>]
```

- `-m`, `--model_path`: Path to the `.npz` file generated during training. Defaults to `output/dense_mnist.npz`.

*Note: The application automatically detects the model architecture (CNN or Dense) and adjusts the input shape accordingly.*

### Example

**Running with default (Dense model):**

```bash
python3 run_gui.py
```

**Running with the Super CNN:**

```bash
python3 run_gui.py -m output/super_cnn_mnist.npz
```

### GUI

- **Draw:** Use your mouse to draw a digit (0-9) on the canvas.
- **Real-time Prediction:** The bar chart on the right updates automatically as you draw.
- **Correction:** Use the "Eraser" or "Erase all" buttons to correct mistakes.
- **Export:** You can save the processed image (what the network sees) using the "Export image" button (saved under `output/`).

## Credits

- GUI layout inspired by [nikhilkumarsingh](https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06).
