from dcase_models.util.files import load_json
from dcase_models.model.models import SB_CNN, A_CRNN
from dcase_models.data.features import MelSpectrogram, Spectrogram
from dcase_models.data.data_generator import DataGenerator

import os
import numpy as np
import pytest

from test_data_generator import TestDataset

params = load_json('parameters.json')
params_features = params['features']

dataset_path = 'data'
dataset = TestDataset(dataset_path)

n_classes = 10
n_frames_cnn = 64
n_freq_cnn = 128

models = [SB_CNN, A_CRNN]
len_models = {'SB_CNN': 15, 'A_CRNN': 25}


@pytest.mark.parametrize("model_class", models)
def test_create_model(model_class):

    params_model = params['models'][model_class.__name__]

    model_container = model_class(
        model=None, model_path=None, n_classes=n_classes,
        n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn,
        **params_model['model_arguments']
    )

    assert len(model_container.model.layers) == \
        len_models[model_class.__name__]


@pytest.mark.parametrize("model_class", models)
def test_load_model(model_class):

    model_path = os.path.join('models', model_class.__name__)

    model_container = model_class(model=None, model_path=model_path)

    assert len(model_container.model.layers) == \
        len_models[model_class.__name__]


feats = [MelSpectrogram, Spectrogram]


@pytest.mark.parametrize("model_class", models)
@pytest.mark.parametrize("feature_extractor_class", feats)
def test_train_model(model_class, feature_extractor_class):
    feature_extractor = feature_extractor_class(
        sequence_time=params_features['sequence_time'],
        sequence_hop_time=params_features['sequence_hop_time'],
        audio_win=params_features['audio_win'],
        audio_hop=params_features['audio_hop'],
        n_fft=params_features['n_fft'],
        sr=params_features['sr'],
        **params_features[feature_extractor_class.__name__]
    )

    data_generator = DataGenerator(dataset, feature_extractor)
    data_generator.load_data()

    X_train = np.concatenate(data_generator.data['all']['X'], axis=0)
    Y_train = np.concatenate(data_generator.data['all']['Y'], axis=0)

    X_val = data_generator.data['all']['X']
    Y_val = data_generator.data['all']['Y']

    n_classes = Y_train.shape[-1]
    n_frames_cnn = X_train.shape[1]
    n_freq_cnn = X_train.shape[2]
    params_model = params['models'][model_class.__name__]

    exp_folder = './'

    model_container = model_class(
        model=None, model_path=None,
        n_classes=n_classes, n_frames_cnn=n_frames_cnn,
        n_freq_cnn=n_freq_cnn,
        **params_model['model_arguments'])

    model_container.train(X_train, Y_train, X_val, Y_val,
                          weights_path=exp_folder, epochs=10)

    model_container.load_model_weights(exp_folder)

    results = model_container.evaluate(X_val, Y_val)

    os.remove('best_weights.hdf5')
    os.remove('training.log')

    assert results['accuracy'] > 0.1
