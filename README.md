<pre>
  ____   ____    _    ____  _____                          _      _     
 |  _ \ / ___|  / \  / ___|| ____|     _ __ ___   ___   __| | ___| |___ 
 | | | | |     / _ \ \___ \|  _| _____| '_ ` _ \ / _ \ / _` |/ _ \ / __|
 | |_| | |___ / ___ \ ___) | |__|_____| | | | | | (_) | (_| |  __/ \__ \
 |____/ \____/_/   \_\____/|_____|    |_| |_| |_|\___/ \__,_|\___|_|___/
                                                                       
</pre>

![example workflow](https://github.com/MTG/DCASE-models/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/MTG/DCASE-models/branch/master/graph/badge.svg?token=xOOVldiH0J)](https://codecov.io/gh/MTG/DCASE-models)
[![PyPI](https://img.shields.io/pypi/v/DCASE-models)](https://pypi.org/project/DCASE-models/)
[![GitHub license](https://img.shields.io/github/license/pzinemanas/DCASE-models)](https://github.com/pzinemanas/DCASE-models/blob/master/LICENSE)


`DCASE-models` is an open-source Python library for rapid prototyping of environmental sound analysis systems, with an emphasis on deep–learning models. The library has a flat and light design that allows easy extension and integration with other existing tools. 

Documentation
-------------
See [https://dcase-models.readthedocs.io](https://dcase-models.readthedocs.io/en/latest/) for a complete reference manual and introductory tutorials.

## Installation instructions
We recommend to install DCASE-models in a dedicated virtual environment. For instance, using [anaconda](https://www.anaconda.com/):
```
conda create -n dcase python=3.6
conda activate dcase
```
For GPU support:
```
conda install cudatoolkit cudnn
```
DCASE-models uses [SoX](http://sox.sourceforge.net/) for functions related to the datasets. You can install it in your conda environment by:
```
conda install -c conda-forge sox
```
When installing the library, you must select the tensorflow variant: version 1 (CPU-only or GPU) or version 2.
``` 
pip install DCASE-models[keras_tf] # for tensorflow 1 CPU-only version
pip install DCASE-models[keras_tf_gpu] # for tensorflow 1 GPU version
pip install DCASE-models[tf2] # for tensorflow 2
```

To include visualization related dependencies, run the following instead:
```
pip install DCASE-models[visualization]
```

## Usage
There are several ways to use this library. In this repository, we accompany the library with three types of examples.

> Note that the default parameters for each model, dataset and feature representation, are stored in [`parameters.json`](parameters.json) on the root directory.

### Python scripts
The folder [`scripts`](scripts) includes python scripts for data downloading, feature extraction, model training and testing, and fine-tuning. These examples show how to use DCASE-models within a python script.

### Jupyter notebooks
The folder [`notebooks`](notebooks) includes a list of notebooks that replicate scientific experiments using DCASE-models.

### Web applications
The folder [`visualization`](visualization) includes a user interface to define, train and visualize the models defined in this library.

Go to DCASE-models folder and run:
```
python -m visualization.index
```
Then, open your browser and navigate to:
```
http://localhost:8050/
```
