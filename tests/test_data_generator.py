import sys
import os
import numpy as np
import argparse

sys.path.append('../')
from dcase_models.utils.files import load_json
from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.data.scaler import Scaler


parser = argparse.ArgumentParser(description='Test DataGenerator')
parser.add_argument('-d', '--dataset', type=str, help='dataset to use for the test', default='UrbanSound8k')
parser.add_argument('-f', '--fold', type=str, help='fold of the dataset', default='fold1')
parser.add_argument('-feat', '--features', type=str, help='features to use for the test', default='MelSpectrogram')
args = parser.parse_args()

params = load_json('parameters.json')
params_dataset = params["datasets"][args.dataset]
params_features = params["features"]

# get feature extractor class
feature_extractor_class = get_available_features()[args.features]
# init feature extractor
feature_extractor = feature_extractor_class(sequence_time=params_features['sequence_time'], 
                                            sequence_hop_time=params_features['sequence_hop_time'], 
                                            audio_win=params_features['audio_win'], 
                                            audio_hop=params_features['audio_hop'], 
                                            n_fft=params_features['n_fft'], 
                                            sr=params_features['sr'], **params_features[args.features])
# get dataset class
dataset_class = get_available_datasets()[args.dataset]

# init dataset
dataset = dataset_class(params_dataset['dataset_path'])

# init data_generator
kwargs = {}
if args.dataset == 'URBAN_SED':
    kwargs = {'sequence_hop_time': params['features']['sequence_hop_time']}
data_generator = DataGenerator(dataset, feature_extractor, **kwargs)

# extract features if needed
data_generator.extract_features()

# load data
print('Loading data... ')
data_generator.load_data()
X_train, Y_train, X_val, Y_val = data_generator.get_data_for_training(args.fold)
print('Done!')

# print shapes
print(X_train.shape, Y_train.shape, len(X_val), len(Y_val))
print(X_val[0].shape, Y_val[0].shape)

# test scaler
print('Fitting scaler... ')
scaler = Scaler(normalizer='minmax')
scaler.fit(X_train)

print('min and max of train set before the scaler is applied', np.amin(X_train), np.amax(X_train))
X_train = scaler.transform(X_train)
print('min and max of train set after the scaler is applied', np.amin(X_train), np.amax(X_train))
print('Done!')