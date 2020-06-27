from dcase_models.util.files import load_json
from dcase_models.data.features import MelSpectrogram, Spectrogram
from dcase_models.data.dataset_base import Dataset
from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.scaler import Scaler

import os
import numpy as np
import pytest
import glob

params = load_json('parameters.json')
params_features = params['features']

dataset_path = 'data'


class TestDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.fold_list = ["all"]
        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]
        self.audio_path = os.path.join(self.dataset_path, 'audio')

    def generate_file_lists(self):
        """
        Create self.file_lists, a dict that includes a list of files per fold.

        Each dataset has a different way of organizing the files. This
        function defines the dataset structure.

        """
        self.file_lists['all'] = sorted(
            glob.glob(os.path.join(self.audio_path, '*.wav'))
        )

    def get_annotations(self, file_path, features):
        y = np.zeros((len(features), len(self.label_list)))
        class_ix = int(os.path.basename(file_path).split('-')[1])
        y[:, class_ix] = 1
        return y


dataset = TestDataset(dataset_path)
audio_files = ['40722-8-0-7.wav', '147764-4-7-0.wav', '176787-5-0-0.wav']
feats = [Spectrogram, MelSpectrogram]


@pytest.mark.parametrize("feature_extractor_class", feats)
def test_feature_extractor(feature_extractor_class):
    feature_extractor = feature_extractor_class(
        sequence_time=params_features['sequence_time'],
        sequence_hop_time=params_features['sequence_hop_time'],
        audio_win=params_features['audio_win'],
        audio_hop=params_features['audio_hop'],
        n_fft=params_features['n_fft'],
        sr=params_features['sr'],
        **params_features[feature_extractor_class.__name__]
    )

    feature_shape = feature_extractor.get_shape()

    data_generator = DataGenerator(dataset, feature_extractor)
    data_generator.load_data()

    assert len(data_generator.data) > 0

    X_test, Y_test = data_generator.get_data_for_testing('all')

    assert len(X_test) == len(audio_files)
    assert len(Y_test) == len(audio_files)

    assert X_test[0].shape[1:] == feature_shape[1:]


@pytest.mark.parametrize("normalizer", ['minmax', 'standard'])
def test_scaler(normalizer):
    X = np.random.randn(100, 32, 128)

    scaler = Scaler(normalizer=normalizer)
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    if normalizer == 'minmax':
        assert (np.amin(X_scaled) == -1.0) & (np.amax(X_scaled) == 1.0)
    elif normalizer == 'standard':
        X_scaled_flat = np.reshape(X, (-1, X.shape[-1]))
        assert X_scaled_flat.shape[1] == X.shape[-1]

        mean = np.mean(X_scaled_flat, axis=0)
        std = np.std(X_scaled_flat, axis=0)

        assert np.allclose(mean, np.zeros(128), rtol=0.1, atol=0.1)
        assert np.allclose(std, np.ones(128), rtol=0.1, atol=0.1)

    X_rec = scaler.inverse_transform(X_scaled)

    assert np.allclose(X_rec, X)
