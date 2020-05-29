import sys
import os
import numpy as np
import argparse

sys.path.append('../')
from dcase_models.utils.files import load_json
from dcase_models.data.data_generator import *
from dcase_models.data.scaler import Scaler
from dcase_models.utils.data import check_features_folder
from dcase_models.data.feature_extractor import FeatureExtractor
from dcase_models.utils.misc import get_class_by_name

parser = argparse.ArgumentParser(description='Test DataGenerator')
parser.add_argument('-d', '--dataset', type=str, help='dataset to use for the test', default='UrbanSound8k')
parser.add_argument('-f', '--fold', type=str, help='fold of the dataset', default='fold1')
args = parser.parse_args()

params = load_json('parameters.json')
params_dataset = params["datasets"][args.dataset]

# extract features if needed
if not check_features_folder(params_dataset['feature_folder'], params["features"]):
    audio_folder = params_dataset['audio_folder']
    feature_folder = params_dataset['feature_folder']
    feature_extractor = FeatureExtractor(**params['features'])
    mkdir_if_not_exists(feature_folder)
    if (args.dataset[:3] == 'ESC') | (args.dataset == 'SONYC_UST'):
        feature_extractor.extract(audio_folder, feature_folder)
    else:
        for fold in params_dataset["fold_list"]:
            print('Extracting features of fold: %s' % fold)
            audio_folder_fold = os.path.join(audio_folder, fold)
            feature_folder_fold = os.path.join(feature_folder, fold)
            feature_extractor.extract(audio_folder_fold, feature_folder_fold)

    feature_extractor.save_parameters_json(os.path.join(feature_folder,'parameters.json'))  

# get dataset class
data_generator_class = get_class_by_name(globals(), args.dataset, DataGenerator)

print(data_generator_class)

kwargs = {}
if args.dataset == 'URBAN_SED':
    kwargs = {'sequence_hop_time': params['features']['sequence_hop_time']}
if args.dataset == 'SONYC_UST':
    kwargs = {'taxonomy_file': params_dataset['taxonomy']}    

data_generator = data_generator_class(params_dataset['audio_folder'], params_dataset['feature_folder'], params_dataset['annotations_folder'], 
                                      'mel_spectrogram', params_dataset['folds'], params_dataset['label_list'], params_dataset['metadata'], **kwargs)

data_generator.load_data()
X_train, Y_train, X_val, Y_val = data_generator.get_data_for_training(args.fold)

print(X_train.shape, Y_train.shape, len(X_val), len(Y_val), X_val[0].shape, Y_val[0].shape)

# test scaler
scaler = Scaler(normalizer='minmax')
scaler.fit(X_train)

print(np.amin(X_train), np.amax(X_train))
X_train = scaler.transform(X_train)
print(np.amin(X_train), np.amax(X_train))