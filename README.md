# Fault Tolerant Control Using Reinforcement Learning

Using reinforcement learning to control hybrid systems under degradation.

## Project structure

* `systems/`: Various reinforcement learning environments for testing.
* `python-envs/`: Directory containing environment files for development/deployment with/without GPU packages.
* Notebooks:
  * E-MAML-*: Code pertaining to the AAAI 2020 submission on enhanced-meta learning.

## Usage

This repository depends on [Anaconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

1. Install external dependencies:

    * Windows

        * [Microsoft Visual Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019) this is needed to compile the `Box2d-py` package needed for some OpenAI gym environments.

        * [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467) enables multi-processing and is used by some `stable-baselines` algorithms.

    * Linux
    
        * Prerequisites for stable-baselines library. Run: `sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev`

2. Install python dependencies

Anaconda environment files are located in the `python-envs/` directory. Files with suffix `_cpu` install libraries without GPU acceleration. Files with prefix `dev` do not install a couple of packages that I authored. Instead those packages should be placed in the same directory as this repository.

```
# The dev_*.yml environment files assume other packages written by the author are already
# in PYTHONPATH. In the notebooks their paths are manually added.

cd python-envs
conda env create -f dev.yml             # GPU support for pyTorch/tensorflow
conda env create -f dev_cpu.yml
```

3. Activate environment

```
conda activate rl
```

4. Run notebooks

```
jupyter notebook
```