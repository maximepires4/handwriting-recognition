# Real-Time Handwriting Recognition 
A GUI application that recognizes handwritten digits in real-time.

**This project serves as a proof-of-concept for [MPNeuralNetwork](https://github.com/maximepires4/mp-neural-network/), my custom Deep Learning library built from scratch.**

<center><img src="images/example.gif" width="75%" height="75%"></center>

## Overview
The goal of this project is to demonstrate that a neural network built without frameworks (like TensorFlow or PyTorch) can perform efficiently in a production-like environment.

The application features a drawing canvas where the user inputs a digit.
The app processes the image on the fly and feeds it to a neural network trained on the MNIST dataset.

## Model & Architecture
The "brain" of this application is a Feed-Forward Neural Network built with `mp-neural-network`.

- **Architecture:** `Input(784) -> Dense(128) -> Tanh -> Dense(40) -> Tanh -> Dense(10) -> Softmax`
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** SGD
- **Accuracy:** 93% on the MNIST test set (10,000 images)

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

You need to install the custom library and the GUI dependencies.

```
pip install -r requirements.txt
```

*Note for Linux users: You might need to install Tkinter manually:* `sudo apt install python3-tk`

### 2. Train a model

```
python create_model.py
```

### 3. Run the application

```
python3 main.py
```

## Project Structure
- `main.py`: Entry point of the application.
- `create_model.py`: Script to train the model using `mpneuralnetwork` and save it using `dill` under `output/`.
- `evaluate.py`: Script to calculate accuracy on the test set.
- `src/`: Contains the logic for the GUI (`paint.py`), image processing (`imagehandler.py`), and model inference (`neuralnethandler.py`).

## Usage

- **Draw:** Use your mouse to draw a digit (0-9) on the canvas.
- **Real-time Prediction:** The bar chart on the right updates automatically as you draw (throttled for performance).
- **Correction:** Use the "Eraser" or "Erase all" buttons to correct mistakes.
- **Export:** You can view the result of the image processing pipeline on your drawing using the "Export image" button (saved under `output/`).

## Credits

- GUI layout inspired by [nikhilkumarsingh](https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06).
