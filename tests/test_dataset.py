from dcase_models.data.dataset_base import Dataset

import os
import numpy as np
import pytest
import shutil
import soundfile as sf


audio_files = ['40722-8-0-7.wav', '147764-4-7-0.wav', '176787-5-0-0.wav']


@pytest.mark.parametrize("sr", [22050, 8000])
def test_change_sampling_rate(sr):
    dataset_path = 'data'
    dataset = Dataset(dataset_path)
    audio_path = dataset.get_audio_paths()
    audio_path_sr = dataset.get_audio_paths(sr)
    if os.path.exists(audio_path_sr):
        shutil.rmtree(audio_path_sr)
    dataset.change_sampling_rate(sr)
    for file_audio in audio_files:
        file_path = os.path.join(audio_path_sr, file_audio)
        file_data, file_sr = sf.read(file_path)
        length_seconds = len(file_data)/float(file_sr)

        file_path_original = os.path.join(audio_path, file_audio)
        file_data_original, file_sr_original = sf.read(file_path_original)
        length_seconds_original = len(file_data_original) / \
            float(file_sr_original)

        assert np.allclose(length_seconds, length_seconds_original,
                           rtol=0.0001, atol=0.0001)

    assert dataset.check_sampling_rate(sr)
