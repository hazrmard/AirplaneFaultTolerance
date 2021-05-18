# Fault Tolerant Control Using Reinforcement Learning

Using reinforcement learning to control hybrid systems under degradation.

## Project structure

* `systems/`: Various reinforcement learning environments for testing.
* `python-envs/`: Directory containing environment files for development/deployment with/without GPU packages.
* Notebooks:
  * \*-MAML-\*: Variations on Model Agnostic Meta-Learning
  * Notebooks with conference name contain code for a published work. They should be run with the appropriate tag/commit referenced in the paper (see below).
  * Scratch-*: Experimentation.

  ### Notebooks x tags/branches for submissions

  For each branch or tag, the listed notebook containes the relevant code for the appropriate conference submission.

  * [Branch: aaai2021](https://git.isis.vanderbilt.edu/ahmedi/airplanefaulttolerance/-/tree/aaai2021) for `E-MAML-*.ipynb`
  * [Branch: phm2020](https://git.isis.vanderbilt.edu/ahmedi/airplanefaulttolerance/-/tree/phm2020 for `Meta-RL.ipynb`
  * [Branch: ifac2020](https://git.isis.vanderbilt.edu/ahmedi/airplanefaulttolerance/-/tree/ifac2020) for `IFAC Congress 2020.ipynb` 

## Usage

This repository depends on [Anaconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

1. Install external dependencies:

    * Windows

        * [Microsoft Visual Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019) this is needed to compile the `Box2d-py` package needed for some OpenAI gym environments.

        * [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467) enables multi-processing and is used by some `stable-baselines` algorithms.

    * Linux
    
        * Optional: prerequisites for stable-baselines library. Run: `sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev`

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