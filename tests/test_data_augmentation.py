from dcase_models.data.data_augmentation import AugmentedDataset, WhiteNoise
from dcase_models.data.dataset_base import Dataset

import os
import shutil
import soundfile as sf
import numpy as np
import glob


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
audio_files = ["147764-4-7-0.wav", "176787-5-0-0.wav", "40722-8-0-7.wav"]


def test_white_noise():
    wn = WhiteNoise(60)
    file_origin = audio_path = os.path.join(dataset_path, "audio/147764-4-7-0.wav")
    file_destination = audio_path = os.path.join(dataset_path, "audio/temp.wav")
    _clean(file_destination)
    wn.build(file_origin, file_destination)
    assert os.path.exists(file_destination)
    data, sr = sf.read(file_destination)
    data_orig, sr_orig = sf.read(file_origin)
    assert sr == sr_orig
    assert len(data_orig) == len(data)
    _clean(file_destination)


def test_data_augmentation():
    augmentations_list = [
        {'type': 'pitch_shift', 'n_semitones': -1},
        {'type': 'time_stretching', 'factor': 1.05},
        {'type': 'white_noise', 'snr': 60}
    ]

    aug_dataset = AugmentedDataset(dataset, 22050, augmentations_list)

    audio_path = os.path.join(dataset_path, "audio22050")
    folders = ['pitch_shift_-1', 'time_stretching_1.05', 'white_noise_60.00']
    for folder in folders:
        _clean(audio_path)

    aug_dataset.process()
    for folder in folders:
        assert os.path.exists(os.path.join(audio_path, folder))
    for file_audio in audio_files:
        file_data, file_sr = sf.read(os.path.join(audio_path, 'original', file_audio))
        # check pitch_shift
        data, sr = sf.read(os.path.join(audio_path, folders[0], file_audio))
        assert sr == file_sr
        assert len(file_data) == len(data)
        # check time_stretching
        data, sr = sf.read(os.path.join(audio_path, folders[1], file_audio))
        assert sr == file_sr
        assert len(file_data) > len(data)
        # check white_noise
        data, sr = sf.read(os.path.join(audio_path, folders[2], file_audio))
        assert sr == file_sr
        assert len(file_data) == len(data)

    _clean(os.path.join(audio_path, folder))

    features = np.zeros((10, 9))
    time_resolution = None
    ann = dataset.get_annotations(
        os.path.join(audio_path, 'original', "147764-4-7-0.wav"),
        features,
        time_resolution
    )
    ann_aug = aug_dataset.get_annotations(
        os.path.join(audio_path, folders[0], "147764-4-7-0.wav"),
        features,
        time_resolution
    )
    assert np.allclose(ann, ann_aug)

    dataset.generate_file_lists()
    file_lists = dataset.file_lists.copy()
    aug_dataset.generate_file_lists()
    assert file_lists == aug_dataset.file_lists

    audio_path, subfolders = aug_dataset.get_audio_paths()
    assert audio_path == os.path.join(dataset_path, "audio")
    for folder in folders:
        assert os.path.join(dataset_path, "audio", folder) in subfolders

    audio_path, subfolders = aug_dataset.get_audio_paths(22050)
    assert audio_path == os.path.join(dataset_path, "audio22050")
    for folder in folders:
        assert os.path.join(dataset_path, "audio22050", folder) in subfolders
