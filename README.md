<pre>
  ____   ____    _    ____  _____                          _      _     
 |  _ \ / ___|  / \  / ___|| ____|     _ __ ___   ___   __| | ___| |___ 
 | | | | |     / _ \ \___ \|  _| _____| '_ ` _ \ / _ \ / _` |/ _ \ / __|
 | |_| | |___ / ___ \ ___) | |__|_____| | | | | | (_) | (_| |  __/ \__ \
 |____/ \____/_/   \_\____/|_____|    |_| |_| |_|\___/ \__,_|\___|_|___/
                                                                       
</pre>

# DCASE-models
DCASE-models is a python library that aims to be a general structure to define, train and evaluate models for DCASE related problems. The idea is to have a modular and easy-to-use library to rapid prototype experiments. Regarding this, we create a class based structure for the differente stages related to a classifier of audio signals: feature extractor, data generator, scaler and the model itself. The library uses librosa for feature extraction and keras for the classifier, but given the modularity, the user can use other platforms.

## Installation instructions
We recommend to install DCASE-models in a dedicated virtual environment. For instance, using anaconda:
```
conda create -n dcase python=3.6
conda activate dcase
```
For GPU support:
```
conda install cudatoolkit cudnn
```
DCASE-models uses [SoX](http://sox.sourceforge.net/) for functions related to the datasets. You can install it in your conda environemnt by:
```
conda install -c conda-forge sox
```
Then to install the package:
```
git clone https://github.com/pzinemanas/DCASE-models.git
cd DCASE-models
pip install .
```
To include visualization related dependencies, run the following instead:
```
pip install .[visualization]
```

## Usage
There are several ways to use this library. In this repository there are examples of three types:

### Python scripts
The folder `scripts` includes python scripts for data downloading, feature extraction, model training and testing. These examples show how to use DCASE-models within a python script.

### Jupyter notebooks
The folder `notebooks` includes a list of notebooks that replicate experiments using DCASE-models.

### Web applications
The folder `visualization` includes a user interface to define, train and visualize the models defined in this library.

Go to DCASE-models folder and run:
```
python -m visualization.index
```

