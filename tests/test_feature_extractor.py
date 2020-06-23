from dcase_models.util.files import load_json
from dcase_models.data.features import MelSpectrogram, Spectrogram
from dcase_models.data.dataset_base import Dataset
from dcase_models.data.data_generator import DataGenerator

import os
import numpy as np
import pytest
import shutil


params = load_json('parameters.json')
params_features = params['features']

dataset_path = 'data'
dataset = Dataset(dataset_path)
files = ['40722-8-0-7.npy', '147764-4-7-0.npy', '176787-5-0-0.npy']
feats = [Spectrogram, MelSpectrogram]


@pytest.mark.parametrize("infile", files)
@pytest.mark.parametrize("feature_extractor_class", feats)
def test_feature_extractor(infile, feature_extractor_class):
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
    shutil.rmtree(data_generator.features_path)

    data_generator.extract_features()
    mel_spec_path = os.path.join(data_generator.features_path, infile)
    mel_spec = np.load(mel_spec_path)

    gt_path = os.path.join(
        dataset_path, 'features_gt',
        feature_extractor_class.__name__, infile
    )
    gt = np.load(gt_path)

    assert np.allclose(mel_spec, gt)
