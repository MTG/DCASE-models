from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.model.models import get_available_models
from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.scaler import Scaler
from dcase_models.utils.files import load_json
from dcase_models.utils.files import mkdir_if_not_exists, save_pickle

import argparse
import os

models_path = './models'
mkdir_if_not_exists(models_path)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name',
                        default='UrbanSound8k')
    parser.add_argument('-f', '--features', type=str, help='features name',
                        default='MelSpectrogram')
    parser.add_argument('-m', '--model', type=str, help='model name',
                        default='SB_CNN')
    parser.add_argument('-fold', '--fold_name', type=str, help='fold name',
                        default='fold1')
    args = parser.parse_args()

    if args.dataset not in get_available_datasets():
        raise AttributeError('Dataset not available')

    if args.features not in get_available_features():
        raise AttributeError('Features not available')

    if args.model not in get_available_models():
        raise AttributeError('Model not available')

    # Get parameters
    params = load_json('parameters.json')
    params_dataset = params['datasets'][args.dataset]
    params_features = params['features']
    params_model = params['models'][args.model]

    # Get and init dataset class
    dataset_class = get_available_datasets()[args.dataset]
    dataset = dataset_class(params_dataset['dataset_path'])

    if args.fold_name not in dataset.fold_list:
        raise AttributeError('Fold not available')

    # Get and init feature class
    features_class = get_available_features()[args.features]
    features = features_class(
        sequence_time=params_features['sequence_time'],
        sequence_hop_time=params_features['sequence_hop_time'],
        audio_win=params_features['audio_win'],
        audio_hop=params_features['audio_hop'],
        sr=params_features['sr'], **params_features[args.features]
    )

    # Init data generator
    data_generator = DataGenerator(dataset, features)
    data_generator.load_data()

    # Check if features were extracted
    if not data_generator.check_if_features_extracted():
        print('Extracting features ...')
        data_generator.extract_features()
        print('Done!')

    # Get data and fit scaler
    X_train, Y_train, X_val, Y_val = data_generator.get_data_for_training(
        args.fold_name
    )
    scaler = Scaler(normalizer=params_model['normalizer'])
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    n_frames_cnn = X_train.shape[1]
    n_freq_cnn = X_train.shape[2]
    n_classes = Y_train.shape[1]
    model_class = get_available_models()[args.model]
    model_container = model_class(
        model=None, model_path=None, n_classes=n_classes,
        n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn,
        **params_model['model_arguments']
    )

    model_container.model.summary()

    model_folder = os.path.join(models_path, args.model)
    mkdir_if_not_exists(model_folder)
    model_folder = os.path.join(model_folder, args.dataset)
    mkdir_if_not_exists(model_folder)
    exp_folder = os.path.join(model_folder, args.fold_name)
    mkdir_if_not_exists(exp_folder)

    model_container.save_model_json(model_folder)
    save_pickle(scaler, os.path.join(exp_folder, 'scaler.pickle'))

    model_container.train(X_train, Y_train, X_val, Y_val,
                          weights_path=exp_folder, **params['train'])


if __name__ == "__main__":
    main()
