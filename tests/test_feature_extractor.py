import sys
import os
import numpy as np
from librosa.core import power_to_db
import matplotlib.pyplot as plt


sys.path.append('../')
# only for vscode
from dcase_models.utils.files import load_json
from dcase_models.data.feature_extractor import FeatureExtractor

dataset = 'UrbanSound8k'

params = load_json('parameters.json')
params_dataset = params["datasets"][dataset]

# define a custom feature extractor
def custom_feature(sr=22050, S=None, audio=None):
    print(audio.shape)
    print(S.shape)
    return S[:100]

# append the custom feature extractor to the feature lsit defined in parameters.json
params['features']['features'].append(custom_feature)

# extract features and save files
feature_extractor = FeatureExtractor(**params['features'])

audio_folder = params_dataset['audio_folder']
feature_folder = params_dataset['feature_folder']
feature_extractor.extract('audio/','features/')

# load features files and show them

files = ['features/mel_spectrogram/40722-8-0-7.npy', 'features/mel_spectrogram/147764-4-7-0.npy', 'features/mel_spectrogram/176787-5-0-0.npy']

for i,fi in enumerate(files):
    mel_spec = np.load(fi)
    #spec = np.load('features/spectrograms/40722-8-0-7.npy')

    n_sequences = mel_spec.shape[0]
    print(mel_spec.shape, n_sequences)

    plt.figure()
    for j in range(n_sequences):
        plt.subplot(1, n_sequences, j+1)
        plt.imshow(mel_spec[j].T)
    plt.colorbar()

    plt.savefig('imgs/features'+str(i)+'.png', dpi=300,  bbox_inches='tight', pad_inches=0)
    plt.show()


feature_extractor.save_mel_basis('features/mel_basis.npy')
feature_extractor.save_parameters_json('features/parameters.json')