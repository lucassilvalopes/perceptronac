# perceptronac

Functions useful for (multi-layer) perceptron arithmetic coding.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

- Python >= 3.6
- Pip
- jbigkit (`sudo apt install jbigkit-bin` in ubuntu)

### Installing

With Python 3 and pip installed, preferably in a separate virtual environment, run :
```
pip install -e .
```

## Running the tests

To be able to run the unit tests run :
```
pip install -e .[unit]
```
To run all unit tests, execute the following in the root directory :
```
pytest
```
Running specifically:
```
pytest tests/unit/test_coding3d.py
```
Will verify that the point cloud data is converted properly to the X (matrix of samples) and y (vector of targets) format.

## Using the Package

This package makes available several functions useful for (multi-layer) perceptron arithmetic coding experiments. See the provided scripts for how to use them. 

## Contributors

- Philip Chou
- Ricardo Lopes de Queiroz
- Lucas Silva Lopes
- Tomás Malheiros Borges

## Acknowledgments

- This README was based on https://gist.github.com/PurpleBooth/109311bb0361f32d87a2