from dcase_models.util.files import load_json
from dcase_models.data.features import MelSpectrogram, Spectrogram
from dcase_models.data.dataset_base import Dataset
from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.scaler import Scaler

import os
import numpy as np
import pytest
import glob

params = load_json("parameters.json")
params_features = params["features"]


class TestDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.fold_list = ["all"]
        self.label_list = [
            "air_conditioner",
            "car_horn",
            "children_playing",
            "dog_bark",
            "drilling",
            "engine_idling",
            "gun_shot",
            "jackhammer",
            "siren",
            "street_music",
        ]
        self.audio_path = os.path.join(self.dataset_path, "audio")

    def generate_file_lists(self):
        """
        Create self.file_lists, a dict that includes a list of files per fold.

        Each dataset has a different way of organizing the files. This
        function defines the dataset structure.

        """
        self.file_lists["all"] = sorted(
            glob.glob(os.path.join(self.audio_path, "*.wav"))
        )

    def get_annotations(self, file_path, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        class_ix = int(os.path.basename(file_path).split("-")[1])
        y[:, class_ix] = 1
        return y


dataset_path = "./tests/data"
dataset = TestDataset(dataset_path)


def test_init():
    # minmax
    scaler = Scaler(normalizer="minmax")
    assert scaler.normalizer == ["minmax"]
    assert len(scaler.scaler) == 1
    assert scaler.scaler[0] == []

    # standard
    scaler = Scaler(normalizer="standard")
    assert scaler.normalizer == ["standard"]
    assert len(scaler.scaler) == 1
    assert type(scaler.normalizer) is list
    assert type(scaler.scaler[0]).__name__ == "StandardScaler"


def test_partial_fit():
    # minmax
    X1 = np.random.randn(100, 32, 128)

    scaler = Scaler(normalizer="minmax")
    scaler.partial_fit(X1)
    assert type(scaler.scaler) is list
    assert len(scaler.scaler) == 1
    assert scaler.scaler[0][0] == np.amin(X1)
    assert scaler.scaler[0][1] == np.amax(X1)

    # standard
    X2 = np.random.randn(100, 32, 128)

    scaler = Scaler(normalizer="standard")
    scaler.partial_fit(X2)
    X2_flat = np.reshape(X2, (-1, X2.shape[-1]))
    assert X2_flat.shape[1] == X2.shape[-1]

    mean = np.mean(X2_flat, axis=0)
    var = np.var(X2_flat, axis=0)

    assert type(scaler.scaler) is list
    assert len(scaler.scaler) == 1
    assert np.allclose(mean, scaler.scaler[0].mean_, rtol=0.001, atol=0.001)
    assert np.allclose(var, scaler.scaler[0].var_, rtol=0.001, atol=0.001)

    # list of scalers
    scaler = Scaler(normalizer=["minmax", "standard"])
    X_list = [X1, X2]
    scaler.partial_fit(X_list)

    assert type(scaler.scaler) is list
    assert len(scaler.scaler) == 2
    assert scaler.scaler[0][0] == np.amin(X1)
    assert scaler.scaler[0][1] == np.amax(X1)
    assert np.allclose(mean, scaler.scaler[1].mean_, rtol=0.001, atol=0.001)
    assert np.allclose(var, scaler.scaler[1].var_, rtol=0.001, atol=0.001)


def test_fit():
    # Array, expect same results than partial_fit
    X = np.random.randn(100, 32, 128)
    scaler = Scaler(normalizer="minmax")
    scaler.fit(X)
    assert type(scaler.scaler) is list
    assert len(scaler.scaler) == 1
    assert scaler.scaler[0][0] == np.amin(X)
    assert scaler.scaler[0][1] == np.amax(X)

    # DataGenerator
    feature_extractor = MelSpectrogram()
    feature_extractor.extract(dataset)
    data_generator = DataGenerator(dataset, feature_extractor, folds=["all"])
    scaler = Scaler(normalizer="minmax")
    scaler.fit(data_generator)
    X, _ = data_generator.get_data()
    assert type(scaler.scaler) is list
    assert len(scaler.scaler) == 1
    assert scaler.scaler[0][0] == np.amin(X)
    assert scaler.scaler[0][1] == np.amax(X)


def test_transform():
    # minmax
    X1 = np.random.randn(100, 32, 128)

    scaler = Scaler(normalizer="minmax")
    scaler.fit(X1)
    X1_scaled = scaler.transform(X1)
    assert np.amin(X1_scaled) == -1.0
    assert np.amax(X1_scaled) == 1.0

    # standard
    X2 = np.random.randn(100, 32, 128)

    scaler = Scaler(normalizer="standard")
    scaler.fit(X2)
    X2_scaled = scaler.transform(X2)
    X2_scaled_flat = np.reshape(X2_scaled, (-1, X2.shape[-1]))
    assert X2_scaled_flat.shape[1] == X2.shape[-1]

    mean = np.mean(X2_scaled_flat, axis=0)
    std = np.std(X2_scaled_flat, axis=0)

    assert np.allclose(mean, np.zeros(128), rtol=0.001, atol=0.001)
    assert np.allclose(std, np.ones(128), rtol=0.001, atol=0.001)

    # list of scalers
    scaler = Scaler(normalizer=["minmax", "standard"])
    X_list = [X1, X2]
    scaler.fit(X_list)
    X_list_scaled = scaler.transform(X_list)

    assert type(X_list_scaled) is list
    assert len(X_list_scaled) == 2
    assert np.allclose(X_list_scaled[0], X1_scaled, rtol=0.001, atol=0.001)
    assert np.allclose(X_list_scaled[1], X2_scaled, rtol=0.001, atol=0.001)

    # DataGenerator
    feature_extractor = MelSpectrogram()
    feature_extractor.extract(dataset)
    data_generator = DataGenerator(dataset, feature_extractor, folds=["all"])
    scaler = Scaler(normalizer="minmax")
    scaler.fit(data_generator)
    data_generator.set_scaler(scaler)
    X, _ = data_generator.get_data()
    assert np.amin(X) == -1.0
    assert np.amax(X) == 1.0


def test_inverse_transform():
    pass
