from dcase_models.data.dataset_base import Dataset

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
