from dcase_models.data.features import MelSpectrogram
from dcase_models.data.dataset_base import Dataset
from dcase_models.data.data_generator import DataGenerator

import os
import numpy as np
import glob
import pytest
import shutil


def _clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


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
        print(self.file_lists["all"])

    def get_annotations(self, file_path, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        class_ix = int(os.path.basename(file_path).split("-")[1])
        y[:, class_ix] = 1
        return y


dataset_path = "./tests/data"
dataset = TestDataset(dataset_path)
feature_extractor = MelSpectrogram()
audio_files = ["147764-4-7-0.wav", "176787-5-0-0.wav", "40722-8-0-7.wav"]
features_files = ["147764-4-7-0.npy", "176787-5-0-0.npy", "40722-8-0-7.npy"]
classes = [4, 5, 8]

feature_extractor.extract(dataset)
features_path = feature_extractor.get_features_path(dataset)

X_gt = []
Y_gt = []
for j, ff in enumerate(features_files):
    feat = np.load(os.path.join(features_path, "original", ff))
    X_gt.append(feat)
    ann = np.zeros((len(feat), 10))
    ann[:, classes[j]] = 1
    Y_gt.append(ann)


def test_init():
    # not a Dataset
    with pytest.raises(AttributeError):
        data_generator = DataGenerator("dataset", feature_extractor, folds=["all"])

    # not a downloaded dataset
    download_file = os.path.join(dataset_path, "download.txt")
    _clean(download_file)
    with pytest.raises(AttributeError):
        data_generator = DataGenerator(dataset, feature_extractor, folds=["all"])
    dataset.set_as_downloaded()

    # not a FeatureExtractor
    with pytest.raises(AttributeError):
        data_generator = DataGenerator(dataset, "feature_extractor", folds=["all"])

    # not extracted features
    features_path = feature_extractor.get_features_path(dataset)
    _clean(features_path)
    with pytest.raises(AttributeError):
        data_generator = DataGenerator(dataset, feature_extractor, folds=["all"])
    feature_extractor.extract(dataset)

    # check audio_file_list
    data_generator = DataGenerator(
        dataset, feature_extractor, folds=["all"], shuffle=False
    )
    audio_path = dataset.audio_path
    assert len(data_generator.audio_file_list) == len(audio_files)
    for filename, afl in zip(audio_files, data_generator.audio_file_list):
        assert afl["file_original"] == os.path.join(audio_path, filename)
        assert afl["sub_folder"] == "original"


def test_get_data_batch():
    data_generator = DataGenerator(
        dataset, feature_extractor, folds=["all"], batch_size=2, shuffle=False
    )
    # check length
    assert len(data_generator) == 2

    # Batch 1
    X1, Y1 = data_generator.get_data_batch(0)
    assert X1.shape[0] == len(X_gt[0]) + len(X_gt[1])
    assert X1.shape[0] == Y1.shape[0]
    assert np.allclose(Y1, np.concatenate((Y_gt[0], Y_gt[1]), axis=0))
    assert np.allclose(X1, np.concatenate((X_gt[0], X_gt[1]), axis=0))

    # Batch 2
    X2, Y2 = data_generator.get_data_batch(1)
    assert X2.shape[0] == len(X_gt[2])
    assert X2.shape[0] == Y2.shape[0]
    assert np.allclose(Y2, Y_gt[2])
    assert np.allclose(X2, X_gt[2])

    # Train=False
    data_generator = DataGenerator(
        dataset,
        feature_extractor,
        folds=["all"],
        batch_size=2,
        shuffle=False,
        train=False,
    )
    # check length
    assert len(data_generator) == 2

    # Batch 1
    X1, Y1 = data_generator.get_data_batch(0)
    assert type(X1) is list
    assert type(Y1) is list
    assert len(X1) == 2
    assert len(Y1) == 2
    assert np.allclose(X1[0], X_gt[0])
    assert np.allclose(Y1[0], Y_gt[0])
    assert np.allclose(X1[1], X_gt[1])
    assert np.allclose(Y1[1], Y_gt[1])

    # Batch 2
    X2, Y2 = data_generator.get_data_batch(1)
    assert type(X2) is list
    assert type(Y2) is list
    assert len(X2) == 1
    assert len(Y2) == 1
    assert np.allclose(X2[0], X_gt[2])
    assert np.allclose(Y2[0], Y_gt[2])


def test_get_data():
    # train=True
    data_generator = DataGenerator(
        dataset, feature_extractor, folds=["all"], batch_size=2, shuffle=False
    )

    X, Y = data_generator.get_data()
    assert np.allclose(X, np.concatenate((X_gt[0], X_gt[1], X_gt[2]), axis=0))
    assert np.allclose(Y, np.concatenate((Y_gt[0], Y_gt[1], Y_gt[2]), axis=0))

    # train=False
    data_generator = DataGenerator(
        dataset,
        feature_extractor,
        folds=["all"],
        batch_size=2,
        shuffle=False,
        train=False,
    )

    X, Y = data_generator.get_data()
    assert type(X) is list
    assert type(Y) is list
    assert len(X) == 3
    assert len(Y) == 3
    for j in range(3):
        assert np.allclose(X[j], X_gt[j])
        assert np.allclose(Y[j], Y_gt[j])


def test_get_data_from_file():
    # train=True
    data_generator = DataGenerator(
        dataset, feature_extractor, folds=["all"], batch_size=2, shuffle=False
    )

    for j in range(3):
        X, Y = data_generator.get_data_from_file(j)
        assert np.allclose(X, X_gt[j])
        assert np.allclose(Y, Y_gt[j])


def test_convert_features_path_to_audio_path():
    data_generator = DataGenerator(
        dataset, feature_extractor, folds=["all"], shuffle=False
    )
    features_path = os.path.join(dataset_path, "features")
    audio_path = os.path.join(dataset_path, "audio")
    features_file = os.path.join(features_path, "test.npy")
    audio_file = data_generator.convert_features_path_to_audio_path(
        features_file, features_path
    )
    assert audio_file == os.path.join(audio_path, "test.wav")

    # sr=22050
    sr = 22050
    audio_path = os.path.join(dataset_path, "audio22050")
    features_file = os.path.join(features_path, "test.npy")
    audio_file = data_generator.convert_features_path_to_audio_path(
        features_file, features_path, sr=sr
    )
    assert audio_file == os.path.join(audio_path, "test.wav")

    # list
    features_file = [
        os.path.join(features_path, "test1.npy"),
        os.path.join(features_path, "test2.npy"),
    ]
    audio_path = os.path.join(dataset_path, "audio")
    audio_file = data_generator.convert_features_path_to_audio_path(
        features_file, features_path
    )
    assert type(audio_file) is list
    assert len(audio_file) == 2
    assert audio_file[0] == os.path.join(audio_path, "test1.wav")
    assert audio_file[1] == os.path.join(audio_path, "test2.wav")


def test_convert_audio_path_to_features_path():
    data_generator = DataGenerator(
        dataset, feature_extractor, folds=["all"], shuffle=False
    )
    features_path = os.path.join(dataset_path, "features")
    audio_path = os.path.join(dataset_path, "audio")
    audio_file = os.path.join(audio_path, "test.wav")
    features_file = data_generator.convert_audio_path_to_features_path(
        audio_file, features_path
    )
    assert features_file == os.path.join(features_path, "test.npy")

    # subfolder='original'
    features_file = data_generator.convert_audio_path_to_features_path(
        audio_file, features_path, subfolder="original"
    )
    assert features_file == os.path.join(features_path, "original", "test.npy")

    # list
    audio_file = [
        os.path.join(features_path, "test1.wav"),
        os.path.join(features_path, "test2.wav"),
    ]
    features_file = data_generator.convert_audio_path_to_features_path(
        audio_file, features_path
    )
    assert type(features_file) is list
    assert len(features_file) == 2
    assert features_file[0] == os.path.join(features_path, "test1.npy")
    assert features_file[1] == os.path.join(features_path, "test2.npy")


def test_paths_remove_aug_subfolder():
    data_generator = DataGenerator(
        dataset, feature_extractor, folds=["all"], shuffle=False
    )
    audio_path = os.path.join(dataset_path, "audio")
    audio_file = os.path.join(audio_path, "original", "test.wav")
    new_audio_file = data_generator.paths_remove_aug_subfolder(audio_file)
    assert new_audio_file == os.path.join(audio_path, "test.wav")
