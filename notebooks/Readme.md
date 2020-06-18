# Notebooks

This folder contains [IPython](http://ipython.org/) / [Jupyter](http://jupyter.org/) interactive notebooks to demonstrate `DCASE-models`.

 - [Basics](https://github.com/pzinemanas/DCASE-models/tree/master/notebooks/basics): examples on how to download datasets and extract features.
 - [Challenges](https://github.com/pzinemanas/DCASE-models/tree/master/notebooks/challenges):   DCASE challenge examples.
	 - [[2020 Task 1]](http://dcase.community/challenge2020/task-acoustic-scene-classification)(http://dcase.community/challenge2020/index). *Acoustic scene clasification.*
 -  [Papers](https://github.com/pzinemanas/DCASE-models/tree/master/notebooks/papers): replicating paper results.
	 - [[SB_CNN]](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_cnn-aug-env_ieeespl_2017.pdf) *Deep Convolutional Neural Networks and Data Augmentation For Environmental Sound Classification* J. Salamon and J. P. Bello IEEE Signal Processing Letters, 24(3), pages 279 - 283, 2017. 
	  - [[SB_CNN_SED]](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_scaper_waspaa_2017.pdf) *Scaper: A Library for Soundscape Synthesis and Augmentation* J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello. In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2017. 
	- [[A_CRNN]](https://arxiv.org/pdf/1706.02291.pdf) *Sound event detection using spatial features and convolutional recurrent neural network* S. Adavanne, P. Pertilä, T. Virtanen ICASSP 2017

## Usage
Datasets should be stored on [`/datasets`](https://github.com/pzinemanas/DCASE-models/tree/master/datasets).

Notebooks are designed to be self-contained, but datasets can be downloaded beforehand  as shown on [`/basics/download_and_prepare_datasets.ipynb`](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/basics/download_and_prepare_datasets.ipynb)

Default parameters for each model/dataset are stored in [`parameters.json`](https://github.com/pzinemanas/DCASE-models/blob/master/parameters.json) on the root directory. Notebooks provide examples on how to modify. these parameters.
