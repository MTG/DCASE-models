Tutorial and examples
=====================

This is a tutorial introduction to quickly get you up and running with `DCASE-models`


The package of the library includes a set of examples, organized into three different categories, which illustrate the usefulness of `DCASE-models` for carrying out research experiments or developing applications. These examples can also be used as templates to be adapted for implementing specific DCASE methods. The type of examples provided are:

 - scripts that perform each step in the typical development pipeline of a DCASE task
 - Jupyter Notebooks that replicate some of the experiments reported in the literature 
 - a web interface for sound classification as an example of a high--level application


Example scripts
---------------

A set of Python scripts is provided in the ``examples`` folder of the package. They perform each step in the typical development pipeline of a DCASE task, i.e downloading a dataset, data augmentation, feature extraction, model training, fine-tuning, and model evaluation. Follow the instructions bellow to know how they are used. 

Parameters setting
~~~~~~~~~~~~~~~~~~

First, note that the default parameters are stored in the ``parameters.json`` file at the root folder of the package. You can use other ``parameters.json`` file by passing its path in the ``-p`` (or ``--path``) argument of each script.

Usage information
~~~~~~~~~~~~~~~~~

In the following, we show examples on how to use these scripts for the typical development pipeline step by step. For further usage information please check each script instructions by typing::

    python download_dataset.py --help

Dataset downloading
~~~~~~~~~~~~~~~~~~~

First, let's start by downloading a dataset. For instance, to download the `ESC-50`_ dataset just type::

    python download_dataset.py -d ESC50

.. note::
    Note that by default the dataset will be downloaded to the ``../datasets/ESC50`` folder, following the path set in the ``parameters.json`` file. 
    You can change the path or the ``parameters.json`` file. The datasets available are listed in the :ref:`Datasets <datasets>` section.


Data augmentation
~~~~~~~~~~~~~~~~~

If you wan to use data augmentation techniques on this dataset, you can run the following script::

    python data_augmentation.py -d ESC50

.. note::
    Note that the name and the parameters of each transformation are defined in the ``parameters.json`` file. 
    The augmentations implemented so far are pitch-shifting, time-stretching, and white noise addition.
    Please check the :class:`~dcase_models.data.AugmentedDataset` class for further information. 

Feature extraction
~~~~~~~~~~~~~~~~~~

Now, you can extract the features for each file in the dataset by typing::

    python extract_features.py -d ESC50 -f MelSpectrogram

.. note::
    Note that you have to specify the features name by the ``-f`` argument, in this case :class:`~dcase_models.data.MelSpectrogram`. 
    All the features representations available are listed in the :ref:`Features <features>` section.

Model training
~~~~~~~~~~~~~~

To train a model is also very straightforward. For instance, to train the :class:`~dcase_models.model.SB_CNN` model on the `ESC-50`_ dataset with the :class:`~dcase_models.data.MelSpectrogram` features extracted before just type::

    python train_model.py -d ESC50 -f MelSpectrogram -m SB_CNN -fold fold1

.. note::
   Note that in this case you have to pass the model name and a fold name as an argument, using ``-m`` and ``-fold``, respectively. 
   This fold is considered to be the fold for testing, meaning that it will not be used during training.
   All the implemented models available are listed in the :ref:`Implemented models <implemented_models>` section.

Model evaluation
~~~~~~~~~~~~~~~~

Once the model is trained, you can evaluate the model in the test set by typing::


    python evaluate_model.py -d ESC50 -f MelSpectrogram -m SB_CNN -fold fold1


.. note::
    Note that the fold specified as an argument is the one used for testing. This scripts prints the results that we get from `sed_eval`_ library.

Fine-tuning
~~~~~~~~~~~

Once you have a model trained in some dataset, you can fine-tune the model on another dataset. For instance, to use a pre-trained model on the `ESC-50`_ dataset and fine-tune it on the `MAVD`_ dataset just type::

    python fine_tuning.py -od ESC50 -ofold fold1 -f MelSpectrogram -m SB_CNN -d MAVD -fold test


.. note::
    Note that the information of the original dataset is set by the ``-od`` and ``-ofold`` arguments. Besides, the ``-d`` and ``-fold`` arguments set the new dataset and the test fold, respectively.

.. _ESC-50: https://github.com/karolpiczak/ESC-50
.. _sed_eval: https://tut-arg.github.io/sed_eval/
.. _MAVD: https://doi.org/10.5281/zenodo.3338727
