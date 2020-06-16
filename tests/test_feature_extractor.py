import sys
import os
import numpy as np
import argparse
from librosa.core import power_to_db
import matplotlib.pyplot as plt


sys.path.append('../')
# only for vscode
from dcase_models.utils.files import load_json
from dcase_models.data.features import get_available_features
from dcase_models.data.dataset_base import Dataset
from dcase_models.data.data_generator import DataGenerator

import glob

parser = argparse.ArgumentParser(description='Test DataGenerator')
parser.add_argument('-d', '--dataset', type=str, help='dataset to use for the test', default='UrbanSound8k')
parser.add_argument('-f', '--features', type=str, help='features to use for the test', default='Spectrogram')
args = parser.parse_args()

params = load_json('parameters.json')
params_dataset = params["datasets"][args.dataset]

# extract features and save files
feature_extractor_class = get_available_features()[args.features]
params_features = params['features']
print(params_features[args.features])
print(feature_extractor_class)
feature_extractor = feature_extractor_class(sequence_time=params_features['sequence_time'], 
                                            sequence_hop_time=params_features['sequence_hop_time'], 
                                            audio_win=params_features['audio_win'], 
                                            audio_hop=params_features['audio_hop'], 
                                            n_fft=params_features['n_fft'], 
                                            sr=params_features['sr'], **params_features[args.features])


dataset_path = 'test_dataset'
dataset = Dataset(dataset_path)
data_generator = DataGenerator(dataset, feature_extractor)
data_generator.extract_features()

# load features files and show them
file_names = ['40722-8-0-7.npy', '147764-4-7-0.npy', '176787-5-0-0.npy']
files = [os.path.join(data_generator.features_path, file_name) for file_name in file_names]

for i, fi in enumerate(files):
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


#feature_extractor.save_parameters_json('features')