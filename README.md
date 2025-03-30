# Handwriting recognition

Handwriting recognition model implemented with my library [MPNeuralNetwork](https://github.com/maximepires4/mp-neural-network) with a gui (MNIST database).

## Usage

### Create a model

Create, train and save a model inside `output/model.pkl`.

```bash
python3 create_model.py
```

To modify the model, edit the variable `network` inside `create_model.py`

### Test the model

Launch a gui for manually testing the model

```bash
python3 main.py
```

## Requirements

* tkinter : `sudo apt install python3-tk`

## Credits

User [nikhilkumarsingh](https://github.com/nikhilkumarsingh) for the gui [base code](https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06).
