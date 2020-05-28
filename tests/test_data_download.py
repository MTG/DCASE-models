import sys
import os
import numpy as np
import argparse

sys.path.append('../')
from dcase_models.utils.files import load_json
from dcase_models.data.data_generator import *
from dcase_models.data.scaler import Scaler
from dcase_models.utils.misc import get_class_by_name

parser = argparse.ArgumentParser(description='Test DataGenerator')
parser.add_argument('-d', '--dataset', type=str, help='dataset to use for the test', default='UrbanSound8k')
args = parser.parse_args()

params = load_json('parameters.json')
params_dataset = params["datasets"][args.dataset]

# get model class
data_generator_class = get_class_by_name(globals(), args.dataset, DataGenerator)
print(data_generator_class)

kwargs = {}
if args.dataset == 'URBAN_SED':
    kwargs = {'sequence_hop_time': params['features']['sequence_hop_time']}

data_generator = data_generator_class(params_dataset['audio_folder'], params_dataset['feature_folder'], params_dataset['annotations_folder'], 
                                      'mel_spectrogram', params_dataset['folds'], params_dataset['label_list'], params_dataset['metadata'], **kwargs)

#data_generator.download_dataset('/data/pzinemanas/UrbanSound8K_TEST')
#data_generator.download_dataset(params_dataset['dataset_folder'])


## Test change sampling rate
data_generator.re_sampling('/data/pzinemanas/UrbanSound8K_TEST/audio', 22050)