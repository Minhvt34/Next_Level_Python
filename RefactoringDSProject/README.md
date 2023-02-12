# Data Science project coding and refactoring

This repository contains the example code of a Data Science / MNIST project and a code refactoring version of the "normal" code. Following @ArjanCodes for more next-level python instruction.

## Install requirements

Installing package using pip

```shell
pip install -r requirements.txt
```

Create a new Anaconda/Miniconda environment

```shell
conda create -n <env name> -f conda.yaml
```

## How to run

```shell
python main.py
```

## Tracking

To start Tensorboard
```shell
tensorboard --logdir runs
```

The output will look like:
```shell
TensorBoard 2.6.0 at http://localhost:6006/ (Press CTRL+C to quit)
```