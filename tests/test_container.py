from dcase_models.model.container import ModelContainer, KerasModelContainer
from dcase_models.model.container import PyTorchModelContainer, SklearnModelContainer
from dcase_models.data.data_generator import KerasDataGenerator
from dcase_models.util.files import save_pickle

import os
import numpy as np
import pytest
import shutil

from dcase_models.backend import backends

if 'torch' in backends:
    import torch
    from torch import nn

    class TorchModel(PyTorchModelContainer):
        class Model(nn.Module):
            def __init__(self, params):
                super().__init__()
                self.layer = nn.Linear(10, 2)

            def forward(self, x):
                y = self.layer(x)
                return y

        def __init__(self, model=None, model_path=None,
                     metrics=['classification']):
            super().__init__(model=model, model_path=model_path,
                             model_name='MLP', metrics=metrics)

    torch_container = TorchModel()
else:
    torch = None

if ('tensorflow1' in backends) | ('tensorflow2' in backends):
    import tensorflow as tf

    tensorflow_version = '2' if tf.__version__.split(".")[0] == "2" else '1'

    if tensorflow_version == '2':
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.models import Model
    else:
        from keras.layers import Input, Dense
        from keras.models import Model

    x = Input(shape=(10,), dtype="float32", name="input")
    y = Dense(2)(x)
    model = Model(x, y)
else:
    tensorflow_version = None

if 'sklearn' in backends:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import SGDClassifier
else:
    sklearn = None

def _clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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
@pytest.mark.skipif(tensorflow_version is None, reason="Tensorflow is not installed")
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

@pytest.mark.skipif(tensorflow_version is None, reason="Tensorflow is not installed")
def test_load_model_from_json():
    _clean("./model.json")
    model_container = KerasModelContainer(model)
    model_container.save_model_json("./")
    model_container = KerasModelContainer()
    model_container.load_model_from_json("./")
    assert len(model_container.model.layers) == 2
    _clean("./model.json")

@pytest.mark.skipif(tensorflow_version is None, reason="Tensorflow is not installed")
def test_save_model_from_json():
    _clean("./model.json")
    model_container = KerasModelContainer(model)
    model_container.save_model_json("./")
    assert os.path.exists("./model.json")
    _clean("./model.json")

@pytest.mark.skipif(tensorflow_version is None, reason="Tensorflow is not installed")
def test_save_model_weights():
    weights_file = "./best_weights.hdf5"
    _clean(weights_file)
    model_container = KerasModelContainer(model)
    model_container.save_model_weights("./")
    assert os.path.exists(weights_file)
    _clean(weights_file)

@pytest.mark.skipif(tensorflow_version is None, reason="Tensorflow is not installed")
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

@pytest.mark.skipif(tensorflow_version is None, reason="Tensorflow is not installed")
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

@pytest.mark.skipif(tensorflow_version is None, reason="Tensorflow is not installed")
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


# PyTorchModelContainer
@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_pytorch_init():
    model_container = TorchModel()
    assert len(list(model_container.model.children())) == 1
    assert model_container.model_name == "MLP"
    assert model_container.metrics == ["classification"]

    model_container = PyTorchModelContainer()
    assert len(list(model_container.model.children())) == 0

@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_pytorch_load_model_from_json():
    model_container = PyTorchModelContainer()
    with pytest.raises(NotImplementedError):
        model_container.load_model_from_json("./")

@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_pytorch_save_model_from_json():
    model_container = PyTorchModelContainer()
    with pytest.raises(NotImplementedError):
        model_container.save_model_json("./")

@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_pytorch_save_model_weights():
    weights_file = "./best_weights.pth"
    _clean(weights_file)
    model_container = TorchModel()
    model_container.save_model_weights("./")
    assert os.path.exists(weights_file)
    _clean(weights_file)

@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_pytorch_load_model_weights():
    weights_file = "./best_weights.pth"
    _clean(weights_file)
    model_container = TorchModel()
    model_container.model.layer.weight.data = torch.full((10, 2), 0.5)
    model_container.model.layer.bias.data = torch.full((2, ), 0.5)
    weights = model_container.model.parameters()
    print(list(weights))
    model_container.save_model_weights("./")
    with torch.no_grad():
        model_container.model.layer.weight = nn.Parameter(torch.zeros_like(model_container.model.layer.weight))
        model_container.model.layer.bias = nn.Parameter(torch.zeros_like(model_container.model.layer.bias))
    new_weights = model_container.model.parameters()
    for param1, param2 in zip(weights, new_weights):
        print(param1, param2)
        assert not torch.allclose(param1, param2)

    model_container.load_model_weights("./")
    new_weights = model_container.model.parameters()
    for param1, param2 in zip(weights, new_weights):
        assert torch.allclose(param1, param2)

@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_pytorch_check_if_model_exists():
    model_container = TorchModel()
    model_file = "./best_weights.pth"
    _clean(model_file)
    print(model_container.model)
    model_container.save_model_weights("./")
    assert model_container.check_if_model_exists("./")

    class TorchModel2(PyTorchModelContainer):
        class Model(nn.Module):
            def __init__(self, params):
                super().__init__()
                self.layer = nn.Linear(11, 2)

            def forward(self, x):
                y = self.layer(x)
                return y

        def __init__(self, model=None, model_path=None,
                     metrics=['classification']):
            super().__init__(model=model, model_path=model_path,
                             model_name='MLP', metrics=metrics)

    model_container = TorchModel2()

    assert not model_container.check_if_model_exists("./")

    _clean(model_file)
    assert not model_container.check_if_model_exists("./")

@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_pytorch_train():
    class ToyModel(PyTorchModelContainer):
        class Model(nn.Module):
            def __init__(self, params):
                super().__init__()
                self.layer = nn.Linear(4, 2)
                self.act = nn.Softmax(-1)

            def forward(self, x):
                y = self.layer(x)
                y = self.act(y)
                return y

        def __init__(self, model=None, model_path=None,
                     metrics=['classification'], use_cuda=True):
            super().__init__(model=model, model_path=model_path,
                             model_name='MLP', metrics=metrics, use_cuda=use_cuda)

    model_container = ToyModel(use_cuda=False)

    X_train = np.concatenate((-np.ones((100, 4)), np.ones((100, 4))), axis=0)
    Y_train = np.zeros((200, 2))
    Y_train[:100, 0] = 1
    Y_train[100:, 1] = 1
    X_val = -np.ones((1, 4))
    Y_val = np.zeros((1, 2))
    Y_val[0, 0] = 1

    X_val2 = np.ones((1, 4))
    Y_val2 = np.zeros((1, 2))
    Y_val2[0, 1] = 1

    file_weights = "./best_weights.pth"
    # file_log = "./training.log"
    _clean(file_weights)
    with torch.no_grad():
        model_container.model.layer.weight = nn.Parameter(torch.zeros_like(model_container.model.layer.weight))
        model_container.model.layer.weight[:2, 0] = -0.5
        model_container.model.layer.weight[2:, 1] = 0.5

    model_container.train(
        (X_train, Y_train),
        ([X_val, X_val2], [Y_val, Y_val2]),
        epochs=3,
        label_list=["1", "2"],
    )
    assert os.path.exists(file_weights)
    _clean(file_weights)

    results = model_container.evaluate(
        ([X_val, X_val2], [Y_val, Y_val2]), label_list=["1", "2"]
    )
    assert results["classification"].results()["overall"]["accuracy"] >= 0.0

    # DataGenerator
    class ToyDataGenerator:
        def __init__(self, X_val, Y_val, train=True):
            self.X_val = X_val
            self.Y_val = Y_val
            self.train = train

        def __len__(self):
            return 3

        def get_data_batch(self, index):
            if self.train:
                return X_val, Y_val
            else:
                return [X_val], [Y_val]

        def shuffle_list(self):
            pass

    data_generator = ToyDataGenerator(X_train, Y_train)
    data_generator_val = ToyDataGenerator(X_val, Y_val, train=False)

    model_container = ToyModel(use_cuda=False)

    with torch.no_grad():
        model_container.model.layer.weight = nn.Parameter(torch.zeros_like(model_container.model.layer.weight))
        model_container.model.layer.weight[:2, 0] = -0.5
        model_container.model.layer.weight[2:, 1] = 0.5

    model_container.train(
        data_generator,
        data_generator_val,
        epochs=3,
        batch_size=None,
        label_list=["1", "2"],
    )
    assert os.path.exists(file_weights)
    _clean(file_weights)

    results = model_container.evaluate(
        data_generator_val, label_list=["1", "2"]
    )
    assert results["classification"].results()["overall"]["accuracy"] >= 0.0

    # Other callbacks
    for metric in ["tagging", "sed"]:
        model_container = ToyModel(metrics=[metric])

        file_weights = "./best_weights.pth"
        _clean(file_weights)
        model_container.train(
            data_generator,
            data_generator_val,
            epochs=3,
            label_list=["1", "2"],
        )
        _clean(file_weights)

        results = model_container.evaluate(data_generator_val, label_list=['1', '2'])
        assert results[metric].results()['overall']['f_measure']['f_measure'] >= 0

def test_pytorch_predict():
    class ToyModel(PyTorchModelContainer):
        class Model(nn.Module):
            def __init__(self, params):
                super().__init__()

            def forward(self, x):
                x = 3*x + 2
                return x

        def __init__(self, model=None, model_path=None,
                     metrics=['classification'], use_cuda=True):
            super().__init__(model=model, model_path=model_path,
                             model_name='MLP', metrics=metrics, use_cuda=use_cuda)

    model_container = ToyModel(use_cuda=False)

    x = np.ones((3, 2))
    pred = model_container.predict(x)
    assert np.allclose(pred, x*3 + 2)

    # multi output
    class ToyModel(PyTorchModelContainer):
        class Model(nn.Module):
            def __init__(self, params):
                super().__init__()

            def forward(self, x):
                y1 = 3*x + 2
                y2 = 4*x
                return [y1, y2]

        def __init__(self, model=None, model_path=None,
                     metrics=['classification'], use_cuda=True):
            super().__init__(model=model, model_path=model_path,
                             model_name='MLP', metrics=metrics, use_cuda=use_cuda)

    model_container = ToyModel(use_cuda=False)

    x = np.ones((3, 2))
    pred = model_container.predict(x)
    assert np.allclose(pred[0], x*3 + 2)
    assert np.allclose(pred[1], x*4)

    # multi input
    class ToyModel(PyTorchModelContainer):
        class Model(nn.Module):
            def __init__(self, params):
                super().__init__()

            def forward(self, x1, x2):
                y = x1 + x2
                return y

        def __init__(self, model=None, model_path=None,
                     metrics=['classification'], use_cuda=True):
            super().__init__(model=model, model_path=model_path,
                             model_name='MLP', metrics=metrics, use_cuda=use_cuda)

    model_container = ToyModel(use_cuda=False)

    x1 = np.ones((3, 2))
    x2 = np.ones((3, 2))
    pred = model_container.predict([x1, x2])
    assert np.allclose(pred, x1 + x2)

def test_pytorch_get_number_of_parameters():
    class ToyModel(PyTorchModelContainer):
        class Model(nn.Module):
            def __init__(self, params):
                super().__init__()
                self.layer = nn.Linear(4, 2)
                self.act = nn.Softmax(-1)

            def forward(self, x):
                y = self.layer(x)
                y = self.act(y)
                return y

        def __init__(self, model=None, model_path=None,
                     metrics=['classification'], use_cuda=True):
            super().__init__(model=model, model_path=model_path,
                             model_name='MLP', metrics=metrics, use_cuda=use_cuda)

    model_container = ToyModel(use_cuda=False)

    assert model_container.get_number_of_parameters() == 10

def test_pytorch_cut_network():
    with pytest.raises(NotImplementedError):
        torch_container.cut_network(None)

def test_pytorch_fine_tuning():
    with pytest.raises(NotImplementedError):
        torch_container.fine_tuning(None)

def test_pytorch_get_available_intermediate_outputs():
    with pytest.raises(NotImplementedError):
        torch_container.get_available_intermediate_outputs()

def test_pytorch_get_intermediate_output():
    with pytest.raises(NotImplementedError):
        torch_container.get_intermediate_output(None, None)

def test_pytorch_load_pretrained_model_weights():
    with pytest.raises(NotImplementedError):
        torch_container.load_pretrained_model_weights()

# SklearnModelContainer
@pytest.mark.skipif(sklearn is None, reason="sklearn is not installed")
def test_sklearn_init():
    with pytest.raises(AttributeError):
        model_container = SklearnModelContainer()
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    assert len(model_container.model.get_params()) > 0
    assert model_container.model_name == "SklearnModelContainer"
    assert model_container.metrics == ["classification"]

    model_path = './'
    model_file = os.path.join(model_path, 'model.skl')
    _clean(model_file)
    save_pickle(model, model_file)
    model_container = SklearnModelContainer(model_path=model_path)
    assert len(model_container.model.get_params()) > 0
    _clean(model_file)

@pytest.mark.skipif(torch is None, reason="sklearn is not installed")
def test_sklearn_load_model_from_json():
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    with pytest.raises(NotImplementedError):
        model_container.load_model_from_json("./")

@pytest.mark.skipif(torch is None, reason="sklearn is not installed")
def test_sklearn_save_model_from_json():
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    with pytest.raises(NotImplementedError):
        model_container.save_model_json("./")

@pytest.mark.skipif(torch is None, reason="sklearn is not installed")
def test_sklearn_save_model_weights():
    model_path = './'
    model_file = os.path.join(model_path, 'model.skl')
    _clean(model_file)
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    model_container.save_model_weights(model_path)
    assert os.path.exists(model_file)
    _clean(model_file)

@pytest.mark.skipif(torch is None, reason="sklearn is not installed")
def test_sklearn_load_model_weights():
    model_path = './'
    model_file = os.path.join(model_path, 'model.skl')
    _clean(model_file)
    model = RandomForestClassifier(random_state=0)

    model_container = SklearnModelContainer(model)
    model_container.save_model_weights(model_path)
    model_container.train((np.zeros((2, 2)), np.zeros(2)))
    params = model.get_params()

    model_container = SklearnModelContainer(model_path=model_path)
    new_params = model_container.model.get_params()

    assert len(params) == len(new_params)
    for key, value in params.items():
        assert value == new_params[key]

    _clean(model_file)

@pytest.mark.skipif(torch is None, reason="sklearn is not installed")
def test_sklearn_check_if_model_exists():
    model_path = './'
    model_file = os.path.join(model_path, 'model.skl')
    _clean(model_file)
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)

    model_container.save_model_weights(model_path)
    assert model_container.check_if_model_exists("./")

    model = RandomForestClassifier(n_estimators=10)
    model_container = SklearnModelContainer(model)
    assert not model_container.check_if_model_exists("./")

    model = SVC()
    model_container = SklearnModelContainer(model)
    assert not model_container.check_if_model_exists("./")

    _clean(model_file)
    assert not model_container.check_if_model_exists("./")


@pytest.mark.skipif(torch is None, reason="sklearn is not installed")
def test_sklearn_train():
    model_path = './'
    model_file = os.path.join(model_path, 'model.skl')
    _clean(model_file)

    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)

    X_train = np.concatenate((-np.ones((100, 4)), np.ones((100, 4))), axis=0)
    Y_train = np.zeros((200, ))
    Y_train[:100] = 0
    Y_train[100:] = 1
    X_val1 = -np.ones((1, 4))
    Y_val1 = np.zeros((1, 2))
    Y_val1[0, 0] = 1

    X_val2 = np.ones((1, 4))
    Y_val2 = np.zeros((1, 2))
    Y_val2[0, 1] = 1

    results = model_container.train((X_train, Y_train), ([X_val1, X_val2], [Y_val1, Y_val2]), label_list=['0', '1'])
    assert results.results()["overall"]["accuracy"] >= 0.9

    # DataGenerator
    class ToyDataGenerator:
        def __init__(self, X_val, Y_val, train=True):
            self.X_val = X_val
            self.Y_val = Y_val
            self.train = train

        def __len__(self):
            return 3

        def get_data_batch(self, index):
            if self.train:
                return self.X_val, self.Y_val
            else:
                return [self.X_val], [self.Y_val]

        def shuffle_list(self):
            pass

    data_generator = ToyDataGenerator(X_train, Y_train)
    data_generator_val = ToyDataGenerator(X_val1, Y_val1, train=False)

    # RandomForest does not include partial_fit
    with pytest.raises(AttributeError):
        model_container.train(data_generator, data_generator_val)

    model = SGDClassifier()
    model_container = SklearnModelContainer(model)

    results = model_container.train(data_generator, data_generator_val, label_list=['0', '1'])
    assert results.results()["overall"]["accuracy"] >= 0.9

    with pytest.raises(AttributeError):
        model_container.train(([X_train], Y_train))
    with pytest.raises(AttributeError):
        model_container.train((X_train, [Y_train]))
    X_train = np.zeros((10, 2, 2))
    with pytest.raises(AttributeError):
        model_container.train((X_train, Y_train))
    X_train = np.zeros((10, 2))
    Y_train = np.zeros((10, 2, 2))
    with pytest.raises(AttributeError):
        model_container.train((X_train, Y_train))

    X_train = np.zeros((10, 2, 2))
    Y_train = np.zeros((10, 2))
    data_generator = ToyDataGenerator(X_train, Y_train)
    with pytest.raises(AttributeError):
        model_container.train(data_generator)

    X_train = np.zeros((10, 2))
    Y_train = np.zeros((10, 2, 2))
    data_generator = ToyDataGenerator(X_train, Y_train)
    with pytest.raises(AttributeError):
        model_container.train(data_generator)

    X_train = np.zeros((10, 2))
    Y_train = []
    data_generator = ToyDataGenerator(X_train, Y_train)
    with pytest.raises(AttributeError):
        model_container.train(data_generator)

    X_train = []
    Y_train = np.zeros((10, 2))
    data_generator = ToyDataGenerator(X_train, Y_train)
    with pytest.raises(AttributeError):
        model_container.train(data_generator)

@pytest.mark.skipif(torch is None, reason="sklearn is not installed")
def test_sklearn_evaluate():
    model_path = './'
    model_file = os.path.join(model_path, 'model.skl')
    _clean(model_file)
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model, metrics=['sed'])

    X_train = np.concatenate((-np.ones((100, 4)), np.ones((100, 4))), axis=0)
    Y_train = np.zeros((200, ))
    Y_train[:100] = 0
    Y_train[100:] = 1
    X_val1 = -np.ones((1, 4))
    Y_val1 = np.zeros((1, 2))
    Y_val1[0, 0] = 1

    X_val2 = np.ones((1, 4))
    Y_val2 = np.zeros((1, 2))
    Y_val2[0, 1] = 1

    model_container.train((X_train, Y_train))
    results = model_container.evaluate(([X_val1, X_val2], [Y_val1, Y_val2]), label_list=['0', '1'])
    assert results['sed'].results()["overall"]["f_measure"]["f_measure"] >= 0.0


@pytest.mark.skipif(torch is None, reason="sklearn is not installed")
def test_sklearn_predict():
    model_path = './'
    model_file = os.path.join(model_path, 'model.skl')
    _clean(model_file)
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)

    X_train = np.concatenate((-np.ones((100, 4)), np.ones((100, 4))), axis=0)
    Y_train = np.zeros((200, ))
    Y_train[:100] = 0
    Y_train[100:] = 1
    X_val1 = -np.ones((1, 4))
    Y_val1 = np.zeros((1, 2))
    Y_val1[0, 0] = 1

    model_container.train((X_train, Y_train))
    pred = model_container.predict(X_val1)
    assert pred.shape == (1, 2)

    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)

    X_train = np.concatenate((-np.ones((100, 4)), np.ones((100, 4))), axis=0)
    Y_train = np.zeros((200, 2))
    Y_train[:100, 0] = 1
    Y_train[100:, 1] = 1

    model_container.train((X_train, Y_train))

    pred = model_container.predict(X_val1)
    assert pred.shape == (1, 2)

def test_sklearn_get_number_of_parameters():
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    assert model_container.get_number_of_parameters() == len(model.get_params())

def test_sklearn_cut_network():
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    with pytest.raises(NotImplementedError):
        model_container.cut_network(None)

def test_sklearn_fine_tuning():
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    with pytest.raises(NotImplementedError):
        model_container.fine_tuning(None)

def test_sklearn_get_available_intermediate_outputs():
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    with pytest.raises(NotImplementedError):
        model_container.get_available_intermediate_outputs()

def test_sklearn_get_intermediate_output():
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    with pytest.raises(NotImplementedError):
        model_container.get_intermediate_output(None, None)

def test_sklearn_load_pretrained_model_weights():
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    with pytest.raises(NotImplementedError):
        model_container.load_pretrained_model_weights()

def test_sklearn_build():
    model = RandomForestClassifier()
    model_container = SklearnModelContainer(model)
    with pytest.raises(NotImplementedError):
        model_container.build()
