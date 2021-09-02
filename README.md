# Pypop

`pypop` is a machine learning library for Python 3 focusing on learning the temporal dynamics of complex systems using temporal point processes. It compiles several types models and algorithms to learn the dynamics of diffusion of temporal point processes from observed data.

## Quick setup

To install the library
```
conda create -n env python=3.8
conda activate env
pip install Cython numpy  # Install prerequisites
pip install -e .  # Install internal lib
```

## Structure

The library consists of several modules:
- `models`: Implements several temporal point process models (Wold process, Hawkes process and Quantized-Hawkes process)
- `plotting`: Various plotting utility functions to monitor and assess the performance of the different algorithms
- `simulation`: Implements sampling algorithms to simulate realizations of the various temporal point process models
- `utils`: Utility functions used throughout the lib
- `fitter`: Implement several inference algorithms
- `priors`, `posteriors`: Implements probability distributions for Bayesian inference algorithms
