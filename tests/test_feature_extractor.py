from dcase_models.data.feature_extractor import FeatureExtractor
from dcase_models.data.features import MelSpectrogram, Spectrogram
from dcase_models.data.dataset_base import Dataset
from dcase_models.data.data_generator import DataGenerator
from dcase_models.util.files import load_json, mkdir_if_not_exists

import os
import numpy as np
import pytest
import shutil
import librosa


def _clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


params = load_json("parameters.json")
params_features = params["features"]


def test_load_audio():
    audio_file = "./tests/data/audio/40722-8-0-7.wav"
    sr = 8000
    feature_extractor = FeatureExtractor(sr=8000)
    audio = feature_extractor.load_audio(
        audio_file, mono=True, change_sampling_rate=False
    )
    assert len(audio) == 88200

    audio = feature_extractor.load_audio(
        audio_file, mono=True, change_sampling_rate=True
    )
    assert len(audio) == 32000

    # stereo
    audio = feature_extractor.load_audio(
        audio_file, mono=False, change_sampling_rate=False
    )
    assert len(audio) == 88200  # This file is mono

    audio_file = "./tests/data/audio/147764-4-7-0.wav"
    audio = feature_extractor.load_audio(
        audio_file, mono=False, change_sampling_rate=False
    )
    assert audio.shape[1] == 2


def test_calculate():
    feature_extractor = FeatureExtractor()
    with pytest.raises(NotImplementedError):
        feature_extractor.calculate("file.wav")


def test_set_as_extracted():
    path = "./tests/data/features"
    feature_extractor = FeatureExtractor()
    feature_extractor.set_as_extracted(path)
    json_features = os.path.join(path, "parameters.json")
    assert os.path.exists(json_features)
    parameters_features = load_json(json_features)
    assert len(parameters_features) == 8

    default_parameters = {
        "sequence_time": 1.0,
        "sequence_hop_time": 0.5,
        "audio_hop": 680,
        "audio_win": 1024,
        "sr": 22050,
        "sequence_frames": 32,
        "sequence_hop": 16,
        "features_folder": "features",
    }
    for key in default_parameters.keys():
        assert default_parameters[key] == parameters_features[key]


def test_check_if_extracted_path():
    path = "./tests/data/features"
    feature_extractor = FeatureExtractor()
    feature_extractor.set_as_extracted(path)
    json_features = os.path.join(path, "parameters.json")
    _clean(json_features)
    feature_extractor.set_as_extracted(path)
    assert feature_extractor.check_if_extracted_path(path)


def test_check_if_extracted():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    path = "./tests/data/features/FeatureExtractor/original"
    mkdir_if_not_exists(path, parents=True)
    feature_extractor = FeatureExtractor()
    json_features = os.path.join(path, "parameters.json")
    _clean(json_features)
    feature_extractor.set_as_extracted(path)
    assert feature_extractor.check_if_extracted(dataset)


def test_get_features_path():
    dataset_path = "./tests/data"
    dataset = Dataset(dataset_path)
    feature_extractor = FeatureExtractor()
    path = "./tests/data/features/FeatureExtractor"
    assert feature_extractor.get_features_path(dataset) == path


def test_pad_audio():
    # No sequence slicing
    # sequence_frames = 32
    feature_extractor = FeatureExtractor(sequence_hop_time=-1, pad_mode="constant")
    n_frames = feature_extractor.sequence_frames - 1
    n_samples = n_frames * feature_extractor.audio_hop + feature_extractor.audio_win
    audio = np.ones(n_samples)
    audio_pad = feature_extractor.pad_audio(audio)
    assert (
        len(audio_pad)
        == feature_extractor.sequence_frames * feature_extractor.audio_hop
        + feature_extractor.audio_win
    )

    # No sequence slicing, audio larger than sequence_time
    n_frames = feature_extractor.sequence_frames + 1
    n_samples = n_frames * feature_extractor.audio_hop + feature_extractor.audio_win
    audio = np.ones(n_samples)
    audio_pad = feature_extractor.pad_audio(audio)
    assert (
        len(audio_pad)
        == feature_extractor.sequence_frames * feature_extractor.audio_hop
        + feature_extractor.audio_win
    )

    # Sequence slicing, audio shorter than one sequence
    # sequence_frames = 32
    # sequence_hop = 16
    feature_extractor = FeatureExtractor(pad_mode="constant")
    n_frames = feature_extractor.sequence_frames - 1
    n_samples = n_frames * feature_extractor.audio_hop + feature_extractor.audio_win
    audio = np.ones(n_samples)
    audio_pad = feature_extractor.pad_audio(audio)
    assert (
        len(audio_pad)
        == feature_extractor.sequence_frames * feature_extractor.audio_hop
        + feature_extractor.audio_win
    )

    # Sequence slicing, audio larger than one sequence
    # sequence_frames = 32
    # sequence_hop = 16
    feature_extractor = FeatureExtractor(pad_mode="constant")
    n_frames = feature_extractor.sequence_frames + 1
    n_samples = n_frames * feature_extractor.audio_hop + feature_extractor.audio_win
    audio = np.ones(n_samples)
    audio_pad = feature_extractor.pad_audio(audio)
    assert (
        len(audio_pad)
        == (feature_extractor.sequence_frames + feature_extractor.sequence_hop)
        * feature_extractor.audio_hop
        + feature_extractor.audio_win
    )

    # Sequence slicing, audio length equal to two sequences
    feature_extractor = FeatureExtractor(pad_mode="constant")
    n_frames = 2 * feature_extractor.sequence_frames
    n_samples = n_frames * feature_extractor.audio_hop + feature_extractor.audio_win
    audio = np.ones(n_samples)
    audio_pad = feature_extractor.pad_audio(audio)
    assert (
        len(audio_pad)
        == (2 * feature_extractor.sequence_frames) * feature_extractor.audio_hop
        + feature_extractor.audio_win
    )


def test_convert_to_sequences():
    feature_extractor = FeatureExtractor(pad_mode="constant")
    audio_rep = np.zeros((feature_extractor.sequence_frames * 2, 2))  # (64, 2)
    frames = feature_extractor.convert_to_sequences(audio_rep)  # (3, 32, 2)
    assert frames.shape == (3, 32, 2)

    # Check that ignore last samples
    audio_rep = np.zeros((feature_extractor.sequence_frames * 2 + 10, 2))  # (74, 2)
    frames = feature_extractor.convert_to_sequences(audio_rep)  # (3, 32, 2)
    assert frames.shape == (3, 32, 2)

    feature_extractor = FeatureExtractor(sequence_hop_time=-1, pad_mode="constant")
    frames = feature_extractor.convert_to_sequences(audio_rep)  # (1, 74, 2)
    assert frames.shape == (1, 32, 2)

    feature_extractor = FeatureExtractor(
        sequence_hop_time=-1, sequence_time=-1, pad_mode="constant"
    )
    frames = feature_extractor.convert_to_sequences(audio_rep)  # (1, 74, 2)
    assert frames.shape == (1, 74, 2)

    # Test it together with pad_audio
    feature_extractor = FeatureExtractor(pad_mode="constant")
    n_frames = feature_extractor.sequence_frames
    n_samples = n_frames * feature_extractor.audio_hop + feature_extractor.audio_win
    audio = np.zeros(n_samples)
    audio_pad = feature_extractor.pad_audio(audio)
    stft = librosa.core.stft(
        audio_pad,
        n_fft=1024,
        hop_length=feature_extractor.audio_hop,
        win_length=feature_extractor.audio_win,
        center=False,
    )
    spectrogram = np.abs(stft) ** 2
    spectrogram = spectrogram.T
    frames = feature_extractor.convert_to_sequences(spectrogram)
    assert frames.shape == (1, feature_extractor.sequence_frames, 513)

    n_frames = 2 * feature_extractor.sequence_frames
    n_samples = n_frames * feature_extractor.audio_hop + feature_extractor.audio_win
    audio = np.zeros(n_samples)
    audio_pad = feature_extractor.pad_audio(audio)
    stft = librosa.core.stft(
        audio_pad,
        n_fft=1024,
        hop_length=feature_extractor.audio_hop,
        win_length=feature_extractor.audio_win,
        center=False,
    )
    spectrogram = np.abs(stft) ** 2
    spectrogram = spectrogram.T
    frames = feature_extractor.convert_to_sequences(spectrogram)
    assert frames.shape == (3, feature_extractor.sequence_frames, 513)
