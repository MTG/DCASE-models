from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.data.data_generator import DataGenerator
from dcase_models.utils.files import load_json

import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name',
                        default='UrbanSound8k')
    parser.add_argument('-f', '--features', type=str, help='features name',
                        default='MelSpectrogram')
    args = parser.parse_args()

    if args.dataset not in get_available_datasets():
        raise AttributeError('Dataset not available')

    if args.features not in get_available_features():
        raise AttributeError('Features not available')

    # Get parameters
    params = load_json('parameters.json')
    params_dataset = params['datasets'][args.dataset]
    params_features = params['features']

    # Get and init dataset class
    dataset_class = get_available_datasets()[args.dataset]
    dataset = dataset_class(params_dataset['dataset_path'])

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

    # Extract features
    if data_generator.check_if_features_extracted():
        print('%s features were extracted already' % args.dataset)
    else:
        print('Extracting features ...')
        data_generator.extract_features()

    print('Done!')


if __name__ == "__main__":
    main()
