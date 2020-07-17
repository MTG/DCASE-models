Installation instructions
=========================

We recommend to install `DCASE-models` in a dedicated virtual environment. For instance,
using `anaconda <https://www.anaconda.com/>`_::

    conda create -n dcase python=3.6
    conda activate dcase

`DCASE-models` uses `SoX <http://sox.sourceforge.net/>`_ for functions related to the
datasets. You can install it in your conda environment by::

    conda install -c conda-forge sox

Before installing the library, you must install only one of the Tensorflow variants (CPU-only or GPU)::

    pip install "tensorflow<1.14" # for CPU-only version
    pip install "tensorflow-gpu<1.14" # for GPU version

Then, you can install the library through the Python Package Index (PyPI) or from the
source as explained below.

pypi
----
The simplest way to install `DCASE-models` is through the Python Package Index (PyPI).
This will ensure that all required dependencies are fulfilled.
This can be achieved by executing the following command::

    pip install dcase_models

or::

    sudo pip install dcase_models

to install system-wide, or::

    pip install -u dcase_models

to install just for your own user.


.. _install_from_source:

source
------

If you've downloaded the archive manually from the `releases
<https://github.com/pzinemanas/dcase_models/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf dcase_models-VERSION.tar.gz
    cd dcase_models-VERSION/
    python setup.py install

If you intend to develop `DCASE-models` or make changes to the source code, you can
install with `pip install -e` to link to your actively developed source tree::

    tar xzf dcase_models-VERSION.tar.gz
    cd dcase_models-VERSION/
    pip install -e .

Alternately, the latest development version can be installed via pip::

    pip install git+https://github.com/pzinemanas/dcase_models


sox
---

Say something about installing other dependencies such as `sox`. 
