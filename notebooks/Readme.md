<pre>
  ____   ____    _    ____  _____                          _      _     
 |  _ \ / ___|  / \  / ___|| ____|     _ __ ___   ___   __| | ___| |___ 
 | | | | |     / _ \ \___ \|  _| _____| '_ ` _ \ / _ \ / _` |/ _ \ / __|
 | |_| | |___ / ___ \ ___) | |__|_____| | | | | | (_) | (_| |  __/ \__ \
 |____/ \____/_/   \_\____/|_____|    |_| |_| |_|\___/ \__,_|\___|_|___/
                                                                       
</pre>

# Notebooks

This folder contains [IPython](http://ipython.org/) / [Jupyter](http://jupyter.org/) interactive notebooks to demonstrate `DCASE-models`.

 - [Basics](https://github.com/pzinemanas/DCASE-models/tree/master/notebooks/basics): examples on how to perform basic tasks.
 	- [Downloading datasets](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/basics/download_and_prepare_datasets.ipynb)
	- [Feature extraction](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/basics/feature_extraction.ipynb)
	- [Data Augmentation](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/basics/data_augmentation.ipynb)
	- [Training and evaluating a model](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/basics/Model_Train_and_evaluate.ipynb)
- [Challenges](https://github.com/pzinemanas/DCASE-models/tree/master/notebooks/challenges):   DCASE challenge examples.
	 - [2020 Task 1](http://dcase.community/challenge2020/task-acoustic-scene-classification)(http://dcase.community/challenge2020/index). *Acoustic scene clasification.* This notebook also shows how to define a model.
 -  [Papers](https://github.com/pzinemanas/DCASE-models/tree/master/notebooks/papers): replicating paper results.
	 - [SB_CNN](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/papers/SB_CNN%20example.ipynb) *Deep Convolutional Neural Networks and Data Augmentation For Environmental Sound Classification* J. Salamon and J. P. Bello IEEE Signal Processing Letters, 24(3), pages 279 - 283, 2017. This notebook includes  data augmentation.
	  - [SB_CNN_SED](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/papers/SB_CNN_SED.ipynb) *Scaper: A Library for Soundscape Synthesis and Augmentation* J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello. In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2017. 
	- [A_CRNN](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/papers/paper_SalamonBello_IEEE-WASPAA-2017_CNN_SED.ipynb) *Sound event detection using spatial features and convolutional recurrent neural network* S. Adavanne, P. Pertilä, T. Virtanen ICASSP 2017

## Usage
Datasets should be stored on [`/datasets`](https://github.com/pzinemanas/DCASE-models/tree/master/datasets).

Notebooks are designed to be self-contained, but datasets can be downloaded beforehand  as shown on [`/basics/download_and_prepare_datasets.ipynb`](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/basics/download_and_prepare_datasets.ipynb).

Basics notebooks can be run sequentially as a tutorial.

Default parameters for each model/dataset are stored in [`parameters.json`](https://github.com/pzinemanas/DCASE-models/blob/master/parameters.json) on the root directory. Notebooks provide examples on how to modify these parameters.
