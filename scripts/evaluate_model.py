from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.model.models import get_available_models
from dcase_models.data.data_generator import DataGenerator
from dcase_models.utils.files import load_json, load_pickle

import argparse
import os

models_path = './models'


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

    # Get data
    X_test, Y_test = data_generator.get_data_for_testing(args.fold_name)

    # Set paths
    model_folder = os.path.join(models_path, args.model, args.dataset)
    exp_folder = os.path.join(model_folder, args.fold_name)

    # Load scaler
    scaler_file = os.path.join(exp_folder, 'scaler.pickle')
    scaler = load_pickle(scaler_file)

    X_test = scaler.transform(X_test)

    # Load model and best weights
    model_class = get_available_models()[args.model]
    model_container = model_class(model=None, model_path=model_folder)
    model_container.load_model_weights(exp_folder)

    results = model_container.evaluate(X_test, Y_test)
    print('Accuracy in %s is %f' % (args.fold_name, results['accuracy']))


if __name__ == "__main__":
    main()
