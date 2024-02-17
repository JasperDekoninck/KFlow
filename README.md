# KFlow
This project is an attempt to implement neural networks from scratch solely using numpy (and scipy). I started this project solely for educational purposes with no intention whatsoever of making the library useful (which is of course implied by the use of numpy). I based my design on TensorFlow and Keras, although if I were to redo it, I would now base my design more on PyTorch, since this would be a bit easier. 

## Install
First, install [Conda](https://docs.conda.io/projects/miniconda/en/latest/) and then run:

```bash
conda create -n kflow python=3.11
conda activate kflow
pip install -e .
```

This will allow you to import kflow into your Python project.

## Running examples and tests
For the tests, you can just run the respective files:
```bash
python tests/operations.py
python tests/optimizers.py
```
The examples require scikit-learn and tensorflow with keras to download some of the used datasets. You can install those using:
```bash
pip install scikit-learn tensorflow keras
```
