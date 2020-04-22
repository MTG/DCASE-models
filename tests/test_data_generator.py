import sys
import os
import glob
import numpy as np
from librosa.core import power_to_db
import matplotlib.pyplot as plt

sys.path.append('../')
from dcase_models.utils.files import load_json
from dcase_models.data.data_generator import *
from dcase_models.data.scaler import Scaler

dataset = 'UrbanSound8k'

params = load_json('parameters.json')
params_dataset = params["datasets"][dataset]

# get model class
try:
    data_generator_class = globals()[dataset]
except:
    print('Warning: using default ModelContainer')
    data_generator_class = DataGenerator()

print(data_generator_class)
data_generator = data_generator_class(params_dataset['audio_folder'], params_dataset['feature_folder'], 'mel_spectrograms',
                                      params_dataset['folds'], params_dataset['label_list'])

data_generator.load_data()
fold_test = 'fold1'
X_train, Y_train, X_val, Y_val = data_generator.get_data_for_training(fold_test)

print(X_train.shape, Y_train.shape, len(X_val), len(Y_val), X_val[0].shape, Y_val[0].shape)

scaler = Scaler(normalizer='minmax')
scaler.fit(X_train)

print(np.amin(X_train), np.amax(X_train))
X_train = scaler.transform(X_train)
print(np.amin(X_train), np.amax(X_train))