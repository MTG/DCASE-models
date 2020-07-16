# encoding: utf-8
"""
Models
======

ModelContainer
--------------

A ModelContainer defines an interface to standardize the behavior of 
machine learning models. It stores the architecture and the parameters
of the model. It provides methods to train and evaluate the model,
and to save and load its architecture and weights.

.. autosummary::
    :toctree: generated/

    ModelContainer
    KerasModelContainer

.. _implemented_models:

Implemented models
------------------

Each implemented model has its own class that inherits from a specific
ModelContainer, such as KerasModelContainer. 

.. autosummary::
    :toctree: generated/

    MLP
    SB_CNN
    SB_CNN_SED
    A_CRNN
    VGGish
    SMel
    MST
 
"""

from .container import *  # pylint: disable=wildcard-import
from .models import *  # pylint: disable=wildcard-import
