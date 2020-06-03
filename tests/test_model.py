import sys
import os
import glob
import numpy as np
import argparse

sys.path.append('../')
from dcase_models.utils.files import load_json, mkdir_if_not_exists
from dcase_models.data.data_generator import *
from dcase_models.model.container import *
from dcase_models.model.models import *
from dcase_models.data.scaler import Scaler
from dcase_models.data.feature_extractor import *
from dcase_models.utils.misc import get_class_by_name

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Test DataGenerator')
parser.add_argument('-d', '--dataset', type=str, help='dataset to use for the test', default='UrbanSound8k')
parser.add_argument('-m', '--model', type=str, help='model to use for the test', default='SB_CNN')
parser.add_argument('-f', '--fold', type=str, help='fold of the dataset', default='fold1')
parser.add_argument('-feat', '--features', type=str, help='features to use for the test', default='MelSpectrogram')

args = parser.parse_args()

params = load_json('parameters.json')
params_dataset = params["datasets"][args.dataset]
params_features = params["features"]
params_model = params['models'][args.model]

# get feature extractor class
feature_extractor_class = get_class_by_name(globals(), args.features, FeatureExtractor)
# init feature extractor
feature_extractor = feature_extractor_class(sequence_time=params_features['sequence_time'], 
                                            sequence_hop_time=params_features['sequence_hop_time'], 
                                            audio_win=params_features['audio_win'], 
                                            audio_hop=params_features['audio_hop'], 
                                            n_fft=params_features['n_fft'], 
                                            sr=params_features['sr'], **params_features[args.features])
# get dataset class
data_generator_class = get_class_by_name(globals(), args.dataset, DataGenerator)
# init data_generator
kwargs = {}
if args.dataset == 'URBAN_SED':
    kwargs = {'sequence_hop_time': params['features']['sequence_hop_time']}
data_generator = data_generator_class(params_dataset['dataset_path'], params_dataset['feature_folder'], args.features, 
                                      audio_folder=params_dataset['audio_folder'], **kwargs)

# extract features if needed
folders_list = data_generator.get_folder_lists()
for audio_features_paths in folders_list:
    print('Extracting features from folder: ', audio_features_paths['audio'])
    response = feature_extractor.extract(audio_features_paths['audio'], audio_features_paths['features'])
    if response is None:
        print('Features already were calculated, continue...')
    print('Done!')

# load data
print('Loading data... ')
data_generator.load_data()
X_train, Y_train, X_val, Y_val = data_generator.get_data_for_training(args.fold)
print('Done!')

# get model class
model_container_class = get_class_by_name(globals(), args.model, DCASEModelContainer)
print(model_container_class)

fold_test = 'fold1'
X_train, Y_train, X_val, Y_val = data_generator.get_data_for_training(fold_test)
scaler = Scaler(normalizer=params_model['normalizer'])
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

n_frames_cnn = X_train.shape[1]
n_freq_cnn = X_train.shape[2]
n_classes = Y_train.shape[1]
print(n_frames_cnn, n_freq_cnn, n_classes)
model_container = model_container_class(model=None, folder=None, n_classes=n_classes, n_frames_cnn=n_frames_cnn, 
                                        n_freq_cnn=n_freq_cnn, **params_model['model_arguments'])

model_container.model.summary()

# load pre-trained weights
# only for VGGish
if args.model == 'VGGish':
    model_container.load_pretrained_model_weights()
    model_container.fine_tuning(-1, new_number_of_classes=10, new_activation='softmax', freeze_source_model=True)

mkdir_if_not_exists(args.model)
exp_folder = os.path.join(args.model, args.dataset)
mkdir_if_not_exists(exp_folder)
exp_folder_fold = os.path.join(exp_folder, fold_test)
mkdir_if_not_exists(exp_folder_fold)

# save model as json
print('saving model to %s' % exp_folder_fold)
model_container.save_model_json(exp_folder_fold)

# train model
kwargs = params["train"]
train_arguments = params_model['train_arguments']
model_container.train(X_train, Y_train, X_val, Y_val, weights_path=exp_folder_fold,  **train_arguments, **kwargs)

# load best_weights
model_container.load_model_weights(exp_folder_fold)

# test model
X_test, Y_test = data_generator.get_data_for_testing(fold_test)
X_test = scaler.transform(X_test)
results = model_container.evaluate(X_test, Y_test)

print('Accuracy in test fold: %f' % results['accuracy'])