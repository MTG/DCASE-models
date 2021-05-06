from dcase_models.model.models import (
    MLP,
    SB_CNN,
    SB_CNN_SED,
    A_CRNN,
    VGGish,
    SMel,
    MST,
    ConcatenatedModel,
)

import numpy as np
import pytest

from dcase_models.backend import backends

if ('tensorflow1' in backends) | ('tensorflow2' in backends):
    import tensorflow as tf
    tensorflow2 = tf.__version__.split(".")[0] == "2"
else:
    tensorflow = None

try:
    import autopool
except:
    autopool = None


@pytest.mark.skipif(tensorflow is None, reason="TensorFlow is not installed")
def test_mlp():
    model_container = MLP()
    assert len(model_container.model.layers) == 7
    inputs = np.zeros((3, 64, 12))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)

    model_container = MLP(n_frames=None)
    assert len(model_container.model.layers) == 6
    inputs = np.zeros((3, 12))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)

    model_container = MLP(temporal_integration="sum")
    assert len(model_container.model.layers) == 7
    inputs = np.zeros((3, 64, 12))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)

    if (not tensorflow2) & (autopool is not None):
        model_container = MLP(temporal_integration="autopool")
        assert len(model_container.model.layers) == 7
        inputs = np.zeros((3, 64, 12))
        outputs = model_container.model.predict(inputs)
        assert outputs.shape == (3, 10)


@pytest.mark.skipif(tensorflow is None, reason="TensorFlow is not installed")
def test_sb_cnn():
    model_container = SB_CNN()
    assert len(model_container.model.layers) == 15
    inputs = np.zeros((3, 64, 128))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)

    model_container = SB_CNN(n_channels=2)
    assert len(model_container.model.layers) == 15
    inputs = np.zeros((3, 64, 128, 2))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)


@pytest.mark.skipif(tensorflow is None, reason="TensorFlow is not installed")
def test_sb_cnn_sed():
    model_container = SB_CNN_SED()
    assert len(model_container.model.layers) == 15
    inputs = np.zeros((3, 64, 128))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)

    model_container = SB_CNN_SED(large_cnn=True)
    assert len(model_container.model.layers) == 17
    inputs = np.zeros((3, 64, 128))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)


@pytest.mark.skipif(tensorflow is None, reason="TensorFlow is not installed")
def test_a_crnn():
    model_container = A_CRNN()
    assert len(model_container.model.layers) == 25
    inputs = np.zeros((3, 64, 128))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)

    model_container = A_CRNN(n_channels=2)
    assert len(model_container.model.layers) == 25
    inputs = np.zeros((3, 64, 128, 2))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)

    model_container = A_CRNN(sed=True)
    assert len(model_container.model.layers) == 24
    inputs = np.zeros((3, 64, 128))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 64, 10)


@pytest.mark.skipif(tensorflow is None, reason="TensorFlow is not installed")
def test_vggish():
    model_container = VGGish()
    assert len(model_container.model.layers) == 13
    inputs = np.zeros((3, 96, 64))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 512)


@pytest.mark.skipif(tensorflow is None, reason="TensorFlow is not installed")
def test_smel():
    model_container = SMel()
    assert len(model_container.model.layers) == 6
    inputs = np.zeros((3, 64, 1024))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 64, 128)


@pytest.mark.skipif(tensorflow is None, reason="TensorFlow is not installed")
def test_mst():
    model_container = MST()
    assert len(model_container.model.layers) == 11
    inputs = np.zeros((3, 22050))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 44, 128)


@pytest.mark.skipif(tensorflow is None, reason="TensorFlow is not installed")
def test_concatenated_model():
    model_mst = MST()
    model_cnn = SB_CNN_SED(n_frames_cnn=44, n_freq_cnn=128)
    model_container = ConcatenatedModel([model_mst, model_cnn])
    assert len(model_container.model.layers) == 3
    inputs = np.zeros((3, 22050))
    outputs = model_container.model.predict(inputs)
    assert outputs.shape == (3, 10)
