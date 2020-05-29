import sys
import os
import numpy as np
import argparse
from librosa.core import power_to_db
import matplotlib.pyplot as plt


sys.path.append('../')
# only for vscode
from dcase_models.utils.files import load_json
from dcase_models.data.feature_extractor import *
from dcase_models.utils.misc import get_class_by_name

parser = argparse.ArgumentParser(description='Test DataGenerator')
parser.add_argument('-d', '--dataset', type=str, help='dataset to use for the test', default='UrbanSound8k')
parser.add_argument('-f', '--features', type=str, help='features to use for the test', default='Spectrogram')
args = parser.parse_args()

params = load_json('parameters.json')
params_dataset = params["datasets"][args.dataset]

# extract features and save files
feature_extractor_class = get_class_by_name(globals(), args.features, FeatureExtractor)
params_features = params['features']
print(params_features[args.features])
print(feature_extractor_class)
feature_extractor = feature_extractor_class(sequence_time=params_features['sequence_time'], 
                                            sequence_hop_time=params_features['sequence_hop_time'], 
                                            audio_win=params_features['audio_win'], 
                                            audio_hop=params_features['audio_hop'], 
                                            n_fft=params_features['n_fft'], 
                                            sr=params_features['sr'], **params_features[args.features])


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


feature_extractor.save_parameters_json('features/parameters.json')