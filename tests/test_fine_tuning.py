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
from dcase_models.utils.misc import get_class_by_name

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Test DataGenerator')
parser.add_argument('-sd', '--source_dataset', type=str, help='dataset used to train the model', default='UrbanSound8k')
parser.add_argument('-dd', '--destination_dataset', type=str, help='dataset used to train the new model', default='UrbanSound8k')
parser.add_argument('-m', '--model', type=str, help='model to use for the test', default='SB_CNN')
parser.add_argument('-f', '--fold', type=str, help='fold of the dataset', default='fold1')
args = parser.parse_args()

params = load_json('parameters.json')

# get destination dataset class
data_generator_class = get_class_by_name(globals(), args.destination_dataset, DataGenerator)
params_dataset = params["datasets"][args.destination_dataset]
data_generator = data_generator_class(params_dataset['audio_folder'], params_dataset['feature_folder'], params_dataset['annotations_folder'], 
                                      'mel_spectrograms', params_dataset['folds'], params_dataset['label_list'], meta_file=params_dataset['metadata'])
data_generator.load_data()

# get model class
model_container_class = get_class_by_name(globals(), args.model, DCASEModelContainer)
params_model = params["models"][args.model]

# load data
fold_test = 'fold1'
X_train, Y_train, X_val, Y_val = data_generator.get_data_for_training(fold_test)
scaler = Scaler(normalizer=params_model['normalizer'])
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# paths
exp_folder_source = os.path.join(args.model, args.source_dataset)
exp_folder_source_fold = os.path.join(exp_folder_source, fold_test)

exp_folder_destination = os.path.join(args.model, args.source_dataset + '_ft_' + args.destination_dataset)
mkdir_if_not_exists(exp_folder_destination)
exp_folder_destination_fold = os.path.join(exp_folder_destination, fold_test)
mkdir_if_not_exists(exp_folder_destination_fold)

# load source model and weights
model_container = model_container_class(model=None, folder=exp_folder_source_fold)
model_container.load_model_weights(exp_folder_source_fold)
model_container.model.summary()

# fine-tune the model
model_container.fine_tuning(-2, new_number_of_classes=50, new_activation='softmax', freeze_source_model=True)
model_container.model.summary()

# save new model as json
print('saving model to %s' % exp_folder_destination_fold)
model_container.save_model_json(exp_folder_destination_fold)

# train new model
kwargs = params["train"]
train_arguments = params_model['train_arguments']
model_container.train(X_train, Y_train, X_val, Y_val, weights_path=exp_folder_destination_fold,  **train_arguments, **kwargs)

# load best_weights
model_container.load_model_weights(exp_folder_destination_fold)

# test model
X_test, Y_test = data_generator.get_data_for_testing(fold_test)
X_test = scaler.transform(X_test)
results = model_container.evaluate(X_test, Y_test)

print('Accuracy in test fold: %f' % results['accuracy'])