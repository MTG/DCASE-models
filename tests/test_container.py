from dcase_models.model.container import KerasModelContainer
from dcase_models.data.features import MelSpectrogram, Spectrogram
from dcase_models.data.data_generator import DataGenerator

from keras.layers import Input, Dense
from keras.models import Model

import os
import numpy as np
import pytest

def _clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)

x = Input(shape=(10,), dtype='float32', name='input')
y = Dense(2)(x)
model = Model(x, y)


def test_init():
    _clean('./model.json')
    model_container = KerasModelContainer(model)
    assert len(model_container.model.layers) == 2
    assert model_container.model_name == "DCASEModelContainer"
    assert model_container.metrics == ['classification']

    model_container.save_model_json('./')
    model_container = KerasModelContainer(model_path='./')
    assert len(model_container.model.layers) == 2
    _clean('./model.json')

def test_load_model_from_json():
    _clean('./model.json')
    model_container = KerasModelContainer(model)
    model_container.save_model_json('./')
    model_container = KerasModelContainer()
    model_container.load_model_from_json('./')
    assert len(model_container.model.layers) == 2
    _clean('./model.json')

def test_save_model_from_json():
    _clean('./model.json')
    model_container = KerasModelContainer(model)
    model_container.save_model_json('./')
    assert os.path.exists('./model.json')
    _clean('./model.json')

def test_save_model_weights():
    weights_file = './best_weights.hdf5'
    _clean(weights_file)
    model_container = KerasModelContainer(model)
    model_container.save_model_weights('./')
    assert os.path.exists(weights_file)
    _clean(weights_file)

def test_save_model_weights():
    weights_file = './best_weights.hdf5'
    _clean(weights_file)
    model_container = KerasModelContainer(model)
    model_container.save_model_weights('./')
    weights = model_container.model.layers[1].get_weights()
    model_container.model.layers[1].set_weights([np.zeros((10,2)), np.zeros(2)])
    model_container.load_model_weights('./')
    new_weights = model_container.model.layers[1].get_weights()
    assert np.allclose(new_weights[0], weights[0])
    assert np.allclose(new_weights[1], weights[1])
    _clean(weights_file)


def test_check_if_model_exists():
    model_container = KerasModelContainer(model)
    model_file = './model.json'
    _clean(model_file)
    model_container.save_model_json('./')
    assert model_container.check_if_model_exists('./')
    
    x = Input(shape=(11,), dtype='float32', name='input')
    y = Dense(2)(x)
    new_model = Model(x, y)  
    model_container = KerasModelContainer(new_model)
    assert not model_container.check_if_model_exists('./')

    _clean(model_file)
    assert not model_container.check_if_model_exists('./')
