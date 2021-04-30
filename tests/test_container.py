from dcase_models.model.container import ModelContainer, KerasModelContainer
from dcase_models.data.data_generator import KerasDataGenerator

import tensorflow as tf

tensorflow2 = tf.__version__.split(".")[0] == "2"

if tensorflow2:
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
else:
    from keras.layers import Input, Dense
    from keras.models import Model

import os
import numpy as np
import pytest
import shutil


def _clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


x = Input(shape=(10,), dtype="float32", name="input")
y = Dense(2)(x)
model = Model(x, y)


# ModelContainer
def test_model_container():
    model_container = ModelContainer()
    with pytest.raises(NotImplementedError):
        model_container.build()
    with pytest.raises(NotImplementedError):
        model_container.train()
    with pytest.raises(NotImplementedError):
        model_container.evaluate(None, None)
    with pytest.raises(NotImplementedError):
        model_container.save_model_json(None)
    with pytest.raises(NotImplementedError):
        model_container.load_model_from_json(None)
    with pytest.raises(NotImplementedError):
        model_container.save_model_weights(None)
    with pytest.raises(NotImplementedError):
        model_container.load_model_weights(None)
    with pytest.raises(NotImplementedError):
        model_container.get_number_of_parameters()
    with pytest.raises(NotImplementedError):
        model_container.check_if_model_exists(None)
    with pytest.raises(NotImplementedError):
        model_container.get_available_intermediate_outputs()
    with pytest.raises(NotImplementedError):
        model_container.get_intermediate_output(None)


# KerasModelContainer
def test_init():
    _clean("./model.json")
    model_container = KerasModelContainer(model)
    assert len(model_container.model.layers) == 2
    assert model_container.model_name == "DCASEModelContainer"
    assert model_container.metrics == ["classification"]

    model_container.save_model_json("./")
    model_container = KerasModelContainer(model_path="./")
    assert len(model_container.model.layers) == 2
    _clean("./model.json")


def test_load_model_from_json():
    _clean("./model.json")
    model_container = KerasModelContainer(model)
    model_container.save_model_json("./")
    model_container = KerasModelContainer()
    model_container.load_model_from_json("./")
    assert len(model_container.model.layers) == 2
    _clean("./model.json")


def test_save_model_from_json():
    _clean("./model.json")
    model_container = KerasModelContainer(model)
    model_container.save_model_json("./")
    assert os.path.exists("./model.json")
    _clean("./model.json")


def test_save_model_weights():
    weights_file = "./best_weights.hdf5"
    _clean(weights_file)
    model_container = KerasModelContainer(model)
    model_container.save_model_weights("./")
    assert os.path.exists(weights_file)
    _clean(weights_file)


def test_load_model_weights():
    weights_file = "./best_weights.hdf5"
    _clean(weights_file)
    model_container = KerasModelContainer(model)
    model_container.save_model_weights("./")
    weights = model_container.model.layers[1].get_weights()
    model_container.model.layers[1].set_weights([np.zeros((10, 2)), np.zeros(2)])
    model_container.load_model_weights("./")
    new_weights = model_container.model.layers[1].get_weights()
    assert np.allclose(new_weights[0], weights[0])
    assert np.allclose(new_weights[1], weights[1])
    _clean(weights_file)


def test_check_if_model_exists():
    model_container = KerasModelContainer(model)
    model_file = "./model.json"
    _clean(model_file)
    model_container.save_model_json("./")
    assert model_container.check_if_model_exists("./")

    x = Input(shape=(11,), dtype="float32", name="input")
    y = Dense(2)(x)
    new_model = Model(x, y)
    model_container = KerasModelContainer(new_model)
    assert not model_container.check_if_model_exists("./")

    _clean(model_file)
    assert not model_container.check_if_model_exists("./")


def test_train():
    x = Input(shape=(4,), dtype="float32", name="input")
    y = Dense(2)(x)
    new_model = Model(x, y)
    model_container = KerasModelContainer(new_model)

    X_train = np.concatenate((np.zeros((100, 4)), np.ones((100, 4))), axis=0)
    Y_train = np.zeros((200, 2))
    Y_train[:100, 0] = 1
    Y_train[100:, 1] = 1
    X_val = np.zeros((1, 4))
    Y_val = np.zeros((1, 2))
    Y_val[0, 0] = 1

    X_val2 = np.ones((1, 4))
    Y_val2 = np.zeros((1, 2))
    Y_val2[0, 1] = 1

    file_weights = "./best_weights.hdf5"
    file_log = "./training.log"
    _clean(file_weights)
    _clean(file_log)
    model_container.train(
        (X_train, Y_train),
        ([X_val, X_val2], [Y_val, Y_val2]),
        epochs=3,
        label_list=["1", "2"],
    )
    assert os.path.exists(file_weights)
    assert os.path.exists(file_log)
    _clean(file_weights)
    _clean(file_log)

    results = model_container.evaluate(
        ([X_val, X_val2], [Y_val, Y_val2]), label_list=["1", "2"]
    )
    assert results["classification"].results()["overall"]["accuracy"] > 0.25

    # DataGenerator
    class ToyDataGenerator:
        def __init__(self, X_val, Y_val):
            self.X_val = X_val
            self.Y_val = Y_val

        def __len__(self):
            return 3

        def get_data_batch(self, index):
            return X_val, Y_val

        def shuffle_list(self):
            pass

    data_generator = ToyDataGenerator(X_train, Y_train)

    data_generator = KerasDataGenerator(data_generator)

    x = Input(shape=(4,), dtype="float32", name="input")
    y = Dense(2)(x)
    new_model = Model(x, y)
    model_container = KerasModelContainer(new_model)

    model_container.train(
        data_generator,
        ([X_val, X_val2], [Y_val, Y_val2]),
        epochs=3,
        batch_size=None,
        label_list=["1", "2"],
    )
    assert os.path.exists(file_weights)
    assert os.path.exists(file_log)
    _clean(file_weights)
    _clean(file_log)

    results = model_container.evaluate(
        ([X_val, X_val2], [Y_val, Y_val2]), label_list=["1", "2"]
    )
    assert results["classification"].results()["overall"]["accuracy"] > 0.25

    # Other callbacks
    for metric in ["tagging", "sed"]:
        model_container = KerasModelContainer(new_model, metrics=[metric])
        file_weights = "./best_weights.hdf5"
        file_log = "./training.log"
        _clean(file_weights)
        _clean(file_log)
        model_container.train(
            (X_train, Y_train),
            ([X_val, X_val2], [Y_val, Y_val2]),
            epochs=3,
            label_list=["1", "2"],
        )
        # assert os.path.exists(file_weights)
        assert os.path.exists(file_log)
        _clean(file_weights)
        _clean(file_log)

        # results = model_container.evaluate(([X_val, X_val2], [Y_val, Y_val2]), label_list=['1', '2'])
        # assert results['classification'].results()['overall']['accuracy'] > 0.25
