# rnn_spatial

Spring 2022 rotation work with Brinkman Lab, Stony Brook University. 

This repository contains a main program `train_model.py`, which trains a simple RNN model to perform a "dead reckoning" task, where it keeps track of an agent's current position using a sequence of velocity values as inputs.

## Directory structure

- `models`: Directory for storing trained RNN models, along with performance information
- `notebooks`: Jupyter notebooks used to analyze experiment results
- `scripts`: Scripts for training various models
  - `train_model.py`: General-purpose command-line script for training RNNs with specified parameters
- `spatialnet`: Source code used for project, collected in a locally editable Python package
- `test`: Code (notebooks and scripts) used to make sure the `spatialnet` code works as intended

## Setup

This project an installation of Python 3.9, along with the following third-party packages:

- numpy
- scipy
- matplotlib
- jupyterlab
- pytorch
- pyyaml

In addition, the local `spatialnet` package needs to be installed into the environment in order for the scripts and notebooks to be able to properly access the source code. This can be done using `pip` (after installing the third-party packages above):
```console
> pip install -e . 
```
This command uses the `setup.py` file in the repository root to install the `spatialnet` package in "editable mode", which means that any edits made to the source code will be immediately reflected in the environment. Installing the source code as a package instead of just loading the files directly eliminates a lot of of PYTHONPATH-related bugs. 

