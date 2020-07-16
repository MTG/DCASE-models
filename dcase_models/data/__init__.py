# encoding: utf-8
"""
Data
====

.. _datasets:

Datasets
--------

Datasets are implemented as specializations of the base class Dataset.

.. autosummary::
    :toctree: generated/

    Dataset
    UrbanSound8k
    ESC50
    ESC10
    URBAN_SED
    SONYC_UST
    TAUUrbanAcousticScenes2019
    TAUUrbanAcousticScenes2020Mobile
    TUTSoundEvents2017
    FSDKaggle2018
    MAVD
  
.. _features:

Features
--------

Features are implemented as specializations of the base class FeatureExtractor.

.. autosummary::
    :toctree: generated/

    FeatureExtractor
    Spectrogram
    MelSpectrogram
    Openl3
    RawAudio
    FramesAudio
 
Augmentation
------------
.. autosummary::
    :toctree: generated/

    AugmentedDataset
    WhiteNoise

DataGenerator
-------------
.. autosummary::
    :toctree: generated/

    DataGenerator
    KerasDataGenerator

Scaler
------
.. autosummary::
    :toctree: generated/

    Scaler

"""

from .dataset_base import *  # pylint: disable=wildcard-import
from .datasets import *  # pylint: disable=wildcard-import
from .data_generator import *  # pylint: disable=wildcard-import
from .data_augmentation import *  # pylint: disable=wildcard-import
from .feature_extractor import *  # pylint: disable=wildcard-import
from .features import *  # pylint: disable=wildcard-import
from .scaler import *  # pylint: disable=wildcard-import
