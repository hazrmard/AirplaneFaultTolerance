# Airplane Fault Tolerance

Using reinforcement learning to control hybrid systems under degradation. This repository contains code for the following papers:

1. Ahmed, I., Quiñones-Grueiro, M. and Biswas, G. (2020). Fault-Tolerant Control of Degrading Systems with On-Policy Reinforcement Learning. IFAC-PapersOnLine (under review).

## Project structure

* `IFAC Congress 2020`: Jupyter notebook containing code and results for experiments in IFAC 2020 paper.
* `tanks.py`: Definitions of the fuel tanks model and OpenAI `gym` environment classes for use in reinforcement learning.
* `utils.py`: Some function used in the notebook for data transformation, not relevant to theory.
* `plotting.py`: Functions for plotting graphs.
* `environment.yml`: Anaconda environment file for running the notebook.
* `dev.yml`: Environment file I used to write this code. It does not install additional libraries I authored, instead I manually specified their locations on disk.

## Usage

This repository depends on [Anaconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

1. Install dependencies

```
conda env create -f environment.yml
```

2. Activate environment

```
conda activate ifac
```

3. Run notebooks

```
jupyter notebook
```