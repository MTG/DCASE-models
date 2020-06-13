import sys
import os
import numpy as np
import argparse

sys.path.append('../')
from dcase_models.utils.files import load_json
from dcase_models.data.data_generator import *
from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.data.scaler import Scaler
from dcase_models.utils.misc import get_class_by_name

parser = argparse.ArgumentParser(description='Test DataGenerator')
parser.add_argument('-d', '--dataset', type=str, help='dataset to use for the test', default='UrbanSound8k')
args = parser.parse_args()

params = load_json('parameters.json')
params_dataset = params["datasets"][args.dataset]

# get model class
data_generator_class = get_available_datasets()[args.dataset]

data_generator = data_generator_class(params_dataset['dataset_path'], params_dataset['feature_folder'], '', 
                                      audio_folder=params_dataset['audio_folder'])

new_sr = 22050

print(data_generator.check_sampling_rate(new_sr))
data_generator.change_sampling_rate(new_sr)  
print(data_generator.check_sampling_rate(new_sr))


# feature_extractor_class= get_available_features()['MelSpectrogram']
# params_features = params['features']

# feature_extractor = feature_extractor_class(sequence_time=params_features['sequence_time'], 
#                                             sequence_hop_time=params_features['sequence_hop_time'], 
#                                             audio_win=params_features['audio_win'], 
#                                             audio_hop=params_features['audio_hop'], 
#                                             n_fft=params_features['n_fft'], 
#                                             sr=params_features['sr'], **params_features['MelSpectrogram'])


# data_generator.extract_features(feature_extractor)