from dcase_models.data.dataset_base import Dataset
from dcase_models.data.datasets import UrbanSound8k, ESC50, ESC10, URBAN_SED, SONYC_UST
from dcase_models.data.datasets import TAUUrbanAcousticScenes2019, TAUUrbanAcousticScenes2020Mobile
from dcase_models.data.datasets import TUTSoundEvents2017, MAVD, FSDKaggle2018

import os
import numpy as np
import pytest
import shutil
import soundfile as sf


audio_files = ["40722-8-0-7.wav", "147764-4-7-0.wav", "176787-5-0-0.wav"]


def _clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


def test_build():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    assert dataset.audio_path == "./tests/data/audio"
    assert dataset.fold_list == ["fold1", "fold2", "fold3"]
    assert dataset.label_list == ["class1", "class2", "class3"]


def test_generate_file_lists():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    dataset.generate_file_lists()
    assert dataset.file_lists == {"fold1": [], "fold2": [], "fold3": []}


def test_get_annotations():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    with pytest.raises(NotImplementedError):
        dataset.get_annotations("file_path", "features", "time_resolution")


def test_download():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    dataset.set_as_downloaded()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    url = "file:////" + os.path.join(dir_path, "resources")
    files = ["remote.zip"]
    unzip_file = os.path.join(dataset_path, "remote.wav")
    _clean(unzip_file)
    assert not dataset.download(url, files, force_download=False)
    file_log = os.path.join(dataset_path, "download.txt")
    _clean(file_log)
    resp = dataset.download(url, files, force_download=True)
    assert resp
    assert os.path.exists(unzip_file)
    _clean(unzip_file)


def test_set_as_downloaded():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    file_log = os.path.join(dataset_path, "download.txt")
    _clean(file_log)
    dataset.set_as_downloaded()
    assert os.path.exists(file_log)


def test_check_if_downloaded():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    file_log = os.path.join(dataset_path, "download.txt")
    _clean(file_log)
    assert not dataset.check_if_downloaded()
    dataset.set_as_downloaded()
    assert dataset.check_if_downloaded()


def test_get_audio_paths():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    audio_paths = dataset.get_audio_paths()
    assert audio_paths[0] == "./tests/data/audio"
    assert audio_paths[1] == ["./tests/data/audio/original"]

    audio_paths = dataset.get_audio_paths(22050)
    assert audio_paths[0] == "./tests/data/audio22050"
    assert audio_paths[1] == ["./tests/data/audio22050/original"]


@pytest.mark.parametrize("sr", [22050, 8000])
def test_change_sampling_rate(sr):
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    audio_path = dataset.get_audio_paths()[0]
    audio_path_sr = dataset.get_audio_paths(sr)[1][0]
    _clean(audio_path_sr)
    dataset.change_sampling_rate(sr)
    for file_audio in audio_files:
        file_path = os.path.join(audio_path_sr, file_audio)
        file_data, file_sr = sf.read(file_path)
        length_seconds = len(file_data) / float(file_sr)

        file_path_original = os.path.join(audio_path, file_audio)
        file_data_original, file_sr_original = sf.read(file_path_original)
        length_seconds_original = len(file_data_original) / float(file_sr_original)

        assert np.allclose(
            length_seconds, length_seconds_original, rtol=0.0001, atol=0.0001
        )

    assert dataset.check_sampling_rate(sr)


def test_check_sampling_rate():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    sr = 22050
    audio_path_sr = dataset.get_audio_paths(sr)[1][0]
    _clean(audio_path_sr)
    dataset.change_sampling_rate(sr)
    assert dataset.check_sampling_rate(sr)


def test_convert_to_wav():
    dataset_path = "./tests/data_aiff"
    dataset = Dataset(dataset_path)
    audio_path = dataset.get_audio_paths()[0]
    aiff_files = ["40722-8-0-7.aiff", "147764-4-7-0.aiff", "176787-5-0-0.aiff"]
    for wavfile in audio_files:
        wavpath = os.path.join(audio_path, wavfile)
        _clean(wavpath)

    dataset.convert_to_wav()
    for wavfile, aifffile in zip(audio_files, aiff_files):
        wavpath = os.path.join(audio_path, wavfile)
        data, sr = sf.read(wavpath)
        wavpath_orig = os.path.join("./tests/data/audio", wavfile)
        data_orig, sr_orig = sf.read(wavpath_orig)
        assert sr_orig == sr
        assert np.allclose(data_orig, data, rtol=0.0001, atol=0.0001)

        _clean(wavpath)


def test_urbansound8k():
    dataset_path = './tests/resources/datasets/UrbanSound8k'
    dataset = UrbanSound8k(dataset_path)
    audio_files = ["40722-8-0-7.wav", "147764-4-7-0.wav", "176787-5-0-0.wav"]

    assert dataset.audio_path == './tests/resources/datasets/UrbanSound8k/audio'

    assert dataset.fold_list == [
        "fold1", "fold2", "fold3", "fold4", "fold5",
        "fold6", "fold7", "fold8", "fold9", "fold10"]
    assert dataset.label_list == [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

    # generate_file_lists
    dataset.generate_file_lists()
    assert type(dataset.file_lists) is dict
    assert len(dataset.file_lists) == len(dataset.fold_list)
    assert len(dataset.file_lists['fold1']) == 3
    for filename in audio_files:
        assert os.path.join(dataset_path, 'audio/fold1', filename) in dataset.file_lists['fold1']
    for fold in dataset.fold_list[1:]:
        assert len(dataset.file_lists[fold]) == 0

    # get_annotations
    feat = np.zeros((11, 2))
    ann = dataset.get_annotations("40722-8-0-7.wav", feat, None)
    assert ann.shape == (11, 10)
    ann_gt = np.zeros((11, 10))
    ann_gt[:, 8] = 1
    assert np.allclose(ann, ann_gt)

    feat = np.zeros((5, 4))
    ann = dataset.get_annotations("audio/147764-4-7-0.wav", feat, None)
    assert ann.shape == (5, 10)
    ann_gt = np.zeros((5, 10))
    ann_gt[:, 4] = 1
    assert np.allclose(ann, ann_gt)

    feat = np.zeros((5, 4))
    ann = dataset.get_annotations("176787-5-0-0", feat, None)
    assert ann.shape == (5, 10)
    ann_gt = np.zeros((5, 10))
    ann_gt[:, 5] = 1
    assert np.allclose(ann, ann_gt)


def test_esc50():
    dataset_path = './tests/resources/datasets/ESC50'
    dataset = ESC50(dataset_path)
    audio_files = ["1-100032-A-0.wav", "1-100038-A-14.wav", "1-100210-A-36.wav"]

    assert dataset.audio_path == './tests/resources/datasets/ESC50/audio'
    assert len(dataset.label_list) == 50
    assert dataset.label_list[0] == "dog"
    assert dataset.label_list[14] == "chirping_birds"
    assert dataset.label_list[36] == "vacuum_cleaner"

    assert len(dataset.metadata) == 3

    assert dataset.metadata["1-100032-A-0.wav"] == {
        'fold': 'fold1', 'class_ix': 0, 'class_name': "dog", 'esc10': True}
    assert dataset.metadata["1-100038-A-14.wav"] == {
        'fold': 'fold1', 'class_ix': 14, 'class_name': "chirping_birds", 'esc10': False}
    assert dataset.metadata["1-100210-A-36.wav"] == {
        'fold': 'fold1', 'class_ix': 36, 'class_name': "vacuum_cleaner", 'esc10': False}

    # generate_file_lists
    dataset.generate_file_lists()
    assert type(dataset.file_lists) is dict
    assert len(dataset.file_lists) == len(dataset.fold_list)
    assert len(dataset.file_lists['fold1']) == 3
    for filename in audio_files:
        assert os.path.join(dataset_path, 'audio', filename) in dataset.file_lists['fold1']
    for fold in dataset.fold_list[1:]:
        assert len(dataset.file_lists[fold]) == 0

    # get_annotations
    feat = np.zeros((11, 2))
    ann = dataset.get_annotations("1-100032-A-0.wav", feat, None)
    assert ann.shape == (11, 50)
    ann_gt = np.zeros((11, 50))
    ann_gt[:, 0] = 1
    assert np.allclose(ann, ann_gt)

    feat = np.zeros((5, 4))
    ann = dataset.get_annotations("audio/1-100038-A-14.wav", feat, None)
    assert ann.shape == (5, 50)
    ann_gt = np.zeros((5, 50))
    ann_gt[:, 14] = 1
    assert np.allclose(ann, ann_gt)

    feat = np.zeros((5, 4))
    ann = dataset.get_annotations("1-100210-A-36.wav", feat, None)
    assert ann.shape == (5, 50)
    ann_gt = np.zeros((5, 50))
    ann_gt[:, 36] = 1
    assert np.allclose(ann, ann_gt)


def test_esc10():
    dataset_path = './tests/resources/datasets/ESC50'
    dataset = ESC10(dataset_path)
    audio_files = ["1-100032-A-0.wav"]
    assert len(dataset.label_list) == 1
    assert dataset.label_list[0] == "dog"

    assert len(dataset.metadata) == 1

    assert dataset.metadata["1-100032-A-0.wav"] == {
        'fold': 'fold1', 'class_ix': 0, 'class_name': "dog", 'esc10': True}

    # generate_file_lists
    assert type(dataset.file_lists) is dict
    assert len(dataset.file_lists) == len(dataset.fold_list)
    assert len(dataset.file_lists['fold1']) == 1
    for filename in audio_files:
        assert os.path.join(dataset_path, 'audio', filename) in dataset.file_lists['fold1']
    for fold in dataset.fold_list[1:]:
        assert len(dataset.file_lists[fold]) == 0


def test_urban_sed():
    dataset_path = './tests/resources/datasets/URBAN-SED'
    audio_files = ["0.wav", "1.wav", "2.wav"]
    dataset = URBAN_SED(dataset_path)
    assert dataset.audio_path == './tests/resources/datasets/URBAN-SED/audio'
    assert dataset.annotations_folder == './tests/resources/datasets/URBAN-SED/annotations'

    # generate_file_lists
    dataset.generate_file_lists()
    assert type(dataset.file_lists) is dict
    assert len(dataset.file_lists) == len(dataset.fold_list)
    assert len(dataset.file_lists['train']) == 3
    for filename in audio_files:
        assert os.path.join(dataset_path, 'audio/train', filename) in dataset.file_lists['train']
    for fold in dataset.fold_list[1:]:
        assert len(dataset.file_lists[fold]) == 1

    # get annotations
    feat = np.zeros((11, 2))
    file_path = os.path.join(dataset_path, 'audio/train', "0.wav")
    ann = dataset.get_annotations(file_path, feat, 1.0)
    assert ann.shape == (11, 10)
    ann_gt = np.zeros((11, 10))
    ann_gt[0:2, 0] = 1
    ann_gt[0:2, 8] = 1
    ann_gt[2:4, 9] = 1
    assert np.allclose(ann, ann_gt)

    feat = np.zeros((3, 2))
    file_path = os.path.join(dataset_path, 'audio/train', "0.wav")
    ann = dataset.get_annotations(file_path, feat, 1.0)
    assert ann.shape == (3, 10)
    ann_gt = np.zeros((3, 10))
    ann_gt[0:2, 0] = 1
    ann_gt[0:2, 8] = 1
    ann_gt[2, 9] = 1

    assert np.allclose(ann, ann_gt)

def test_sonyc_ust():
    dataset_path = './tests/resources/datasets/SONYC-UST'
    dataset = SONYC_UST(dataset_path)
    assert dataset.audio_path == './tests/resources/datasets/SONYC-UST/audio'

    # generate_file_lists
    dataset.generate_file_lists()
    fold_files = {}
    fold_files['train'] = ["41_020874.wav", "45_006905.wav"]
    fold_files['validate'] = ["00_000118.wav"]
    for fold in dataset.fold_list:
        for filename in fold_files[fold]:
            assert os.path.join(dataset_path, 'audio', filename) in dataset.file_lists[fold]

    # get annotations
    # get annotations
    feat = np.zeros((11, 2))
    file_path = os.path.join(dataset_path, 'audio', "41_020874.wav")
    ann = dataset.get_annotations(file_path, feat, None)
    assert ann.shape == (11, 8)
    ann_gt = np.zeros((11, 8))
    assert np.allclose(ann, ann_gt)

    feat = np.zeros((11, 2))
    file_path = os.path.join(dataset_path, 'audio', "45_006905.wav")
    ann = dataset.get_annotations(file_path, feat, None)
    assert ann.shape == (11, 8)
    ann_gt = np.zeros((11, 8))
    ann_gt[:, [0, 1]] = 1
    assert np.allclose(ann, ann_gt)

    feat = np.zeros((11, 2))
    file_path = os.path.join(dataset_path, 'audio', "00_000118.wav")
    ann = dataset.get_annotations(file_path, feat, None)
    assert ann.shape == (11, 8)
    ann_gt = np.zeros((11, 8))
    ann_gt[:, [0, 6]] = 1
    assert np.allclose(ann, ann_gt)


def test_tau2019():
    dataset_path = './tests/resources/datasets/TAUUrbanAcousticScenes2019'
    dataset = TAUUrbanAcousticScenes2019(dataset_path)
    assert dataset.audio_path == './tests/resources/datasets/TAUUrbanAcousticScenes2019/audio'
    assert dataset.meta_file == './tests/resources/datasets/TAUUrbanAcousticScenes2019/meta.csv'

    # generate_file_lists
    dataset.generate_file_lists()
    assert type(dataset.file_lists) is dict
    assert len(dataset.file_lists) == len(dataset.fold_list)
    assert len(dataset.file_lists['train']) == 3
    assert len(dataset.file_lists['test']) == 3

    train_files = [
        "audio/airport-lisbon-1000-40000-a.wav",
        "audio/bus-lyon-1001-40001-a.wav",
        "audio/shopping_mall-lisbon-1002-40002-a.wav"
    ]

    test_files = [
        "audio/street_pedestrian-lyon-1162-44093-a.wav",
        "audio/metro-prague-1163-44094-a.wav",
        "audio/park-milan-1164-44095-a.wav"
    ]

    for filename in train_files:
        assert os.path.join(dataset_path, filename) in dataset.file_lists['train']
    for filename in test_files:
        assert os.path.join(dataset_path, filename) in dataset.file_lists['test']


    # get_annotations
    feat = np.zeros((11, 2))
    ann = dataset.get_annotations("audio/airport-lisbon-1000-40000-a.wav", feat, None)
    assert ann.shape == (11, 10)
    ann_gt = np.zeros((11, 10))
    ann_gt[:, 0] = 1
    assert np.allclose(ann, ann_gt)

    ann = dataset.get_annotations("audio/bus-lyon-1001-40001-a.wav", feat, None)
    assert ann.shape == (11, 10)
    ann_gt = np.zeros((11, 10))
    ann_gt[:, 7] = 1
    assert np.allclose(ann, ann_gt)


def test_tut2017():
    dataset_path = './tests/resources/datasets/TUTSoundEvents2017'
    dataset = TUTSoundEvents2017(dataset_path)

    assert dataset.audio_path == './tests/resources/datasets/TUTSoundEvents2017/audio'

    fold_files = {}
    fold_files['fold1'] = ["b099.wav", "b008.wav", "b100.wav", "a013.wav", "a010.wav", "a129.wav"]
    fold_files['fold2'] = ["b009.wav", "a124.wav", "b091.wav", "b098.wav", "a003.wav", "b093.wav"]
    fold_files['fold3'] = ["b095.wav", "a008.wav", "a127.wav", "a131.wav", "b003.wav", "b007.wav"]
    fold_files['fold4'] = ["a128.wav", "b006.wav", "a001.wav", "b005.wav", "b094.wav", "a012.wav"]
    fold_files['test'] = ["a011.wav", "b004.wav", "a009.wav", "b092.wav", "a005.wav", "a123.wav", "b002.wav", "a002.wav"]

    # generate_file_lists
    dataset.generate_file_lists()
    assert type(dataset.file_lists) is dict
    assert len(dataset.file_lists) == len(dataset.fold_list) + 1
    for fold in dataset.fold_list + ['test']:
        assert len(dataset.file_lists[fold]) == len(fold_files[fold])
        for filename in fold_files[fold]:
            assert os.path.join(dataset_path, 'audio/street', filename) in dataset.file_lists[fold]

    # get annotations
    feat = np.zeros((15, 2))
    file_path = os.path.join(dataset_path, 'audio/street', "a001.wav")
    ann = dataset.get_annotations(file_path, feat, 1.0)
    assert ann.shape == (15, 6)
    ann_gt = np.zeros((15, 6))
    ann_gt[1:3, 5] = 1
    ann_gt[3:5, 3] = 1
    ann_gt[4:15, 1] = 1
    assert np.allclose(ann, ann_gt)

    feat = np.zeros((15, 2))
    file_path = os.path.join(dataset_path, 'audio/street', "b099.wav")
    ann = dataset.get_annotations(file_path, feat, 1.0)
    assert ann.shape == (15, 6)
    ann_gt = np.zeros((15, 6))
    ann_gt[1:3, 5] = 1
    ann_gt[3:5, 3] = 1
    ann_gt[4:15, 1] = 1
    assert np.allclose(ann, ann_gt)

    feat = np.zeros((3, 2))
    file_path = os.path.join(dataset_path, 'audio/street', "b099.wav")
    ann = dataset.get_annotations(file_path, feat, 1.0)
    assert ann.shape == (3, 6)
    ann_gt = np.zeros((3, 6))
    ann_gt[1:3, 5] = 1
    assert np.allclose(ann, ann_gt)

def test_fsdkaggle2018():
    dataset_path = './tests/resources/datasets/FSDKaggle2018'
    dataset = FSDKaggle2018(dataset_path)
    assert dataset.audio_path == './tests/resources/datasets/FSDKaggle2018/audio'
    assert dataset.meta_path == './tests/resources/datasets/FSDKaggle2018/meta'

    label_list = ["Chime", "Electric_piano", "Hi-hat", "Saxophone", "Trumpet"]
    for label in label_list:
        assert label in dataset.label_list

    # generate_file_lists
    fold_files = {}
    fold_files['train'] = ["00044347.wav", "001ca53d.wav", "002d256b.wav"]
    fold_files['test'] = ["008afd93.wav", "00ae03f6.wav"]
    fold_files['validate'] = ["00eac343.wav"]

    dataset.generate_file_lists()
    assert type(dataset.file_lists) is dict
    assert len(dataset.file_lists) == len(dataset.fold_list)
    for fold in dataset.fold_list:
        assert len(dataset.file_lists[fold]) == len(fold_files[fold])
        for filename in fold_files[fold]:
            if fold == 'validate':
                fold_folder = 'test'
            else:
                fold_folder = fold
            assert os.path.join(dataset_path, 'audio', fold_folder, filename) in dataset.file_lists[fold]

    # get_annotations
    feat = np.zeros((11, 2))
    ann = dataset.get_annotations("audio/00044347.wav", feat, None)
    assert ann.shape == (11, 5)
    ann_gt = np.zeros((11, 5))
    ann_gt[:, 2] = 1
    assert np.allclose(ann, ann_gt)    

def test_mavd():
    dataset_path = './tests/resources/datasets/MAVD'
    dataset = MAVD(dataset_path)
    assert dataset.audio_path == './tests/resources/datasets/MAVD/audio'
    assert dataset.annotations_path == './tests/resources/datasets/MAVD/annotations'

    # generate_file_lists
    audio_files = ['0.wav', '1.wav', '2.wav']
    dataset.generate_file_lists()
    assert type(dataset.file_lists) is dict
    assert len(dataset.file_lists) == len(dataset.fold_list)
    assert len(dataset.file_lists['train']) == 3
    for filename in audio_files:
        assert os.path.join(dataset_path, 'audio/train', filename) in dataset.file_lists['train']
    for fold in dataset.fold_list[1:]:
        assert len(dataset.file_lists[fold]) == 0

    # get annotations
    feat = np.zeros((11, 2))
    file_path = os.path.join(dataset_path, 'audio/train', "0.wav")
    ann = dataset.get_annotations(file_path, feat, 1.0)
    assert ann.shape == (11, 9)
    ann_gt = np.zeros((11, 9))
    ann_gt[0:2, 0] = 1
    ann_gt[0:2, 4] = 1
    ann_gt[2:4, 8] = 1
    assert np.allclose(ann, ann_gt)
