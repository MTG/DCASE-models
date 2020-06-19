r'''
  ____   ____    _    ____  _____                          _      _
 |  _ \ / ___|  / \  / ___|| ____|     _ __ ___   ___   __| | ___| |___
 | | | | |     / _ \ \___ \|  _| _____| '_ ` _ \ / _ \ / _` |/ _ \ / __|
 | |_| | |___ / ___ \ ___) | |__|_____| | | | | | (_) | (_| |  __/ \__ \\
 |____/ \____/_/   \_\____/|_____|    |_| |_| |_|\___/ \__,_|\___|_|___/

 Download dataset example

'''

import os
import argparse

from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.utils.files import load_json
from dcase_models.data.data_augmentation import AugmentedDataset
from dcase_models.data.data_generator import DataGenerator


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-d', '--dataset', type=str,
        help='dataset name (e.g. UrbanSound8k, ESC50, URBAN_SED, SONYC_UST)',
        default='UrbanSound8k'
    )
    parser.add_argument(
        '-f', '--features', type=str,
        help='features name (e.g. Spectrogram, MelSpectrogram, Openl3)',
        default='MelSpectrogram'
    )
    parser.add_argument(
        '-p', '--path', type=str,
        help='path to the parameters.json file',
        default='../'
    )
    args = parser.parse_args()

    print(__doc__)

    if args.dataset not in get_available_datasets():
        raise AttributeError('Dataset not available')

    # Get parameters
    parameters_file = os.path.join(args.path, 'parameters.json')
    params = load_json(parameters_file)
    params_dataset = params['datasets'][args.dataset]
    params_features = params['features']

    # Get and init dataset class
    dataset_class = get_available_datasets()[args.dataset]
    dataset_path = os.path.join(args.path, params_dataset['dataset_path'])
    dataset = dataset_class(dataset_path)

    # Define the augmentations
    augmentations = [
        {'type': 'pitch_shift', 'n_semitones': -1},
        {'type': 'time_stretching', 'factor': 1.05},
    ]

    # Initialize AugmentedDataset
    aug_dataset = AugmentedDataset(
        dataset, params['features']['sr'], augmentations
    )

    # Process all files
    print('Processing ...')
    aug_dataset.process()
    print('Done!')

    # Extract Features
    features_class = get_available_features()[args.features]
    features = features_class(
        sequence_time=params_features['sequence_time'],
        sequence_hop_time=params_features['sequence_hop_time'],
        audio_win=params_features['audio_win'],
        audio_hop=params_features['audio_hop'],
        sr=params_features['sr'], **params_features[args.features]
    )

    # Get data
    data_generator = DataGenerator(aug_dataset, features)
    data_generator.extract_features()
    data_generator.load_data()
    X, Y = data_generator.get_data_for_testing('fold1')
    print(len(X))


if __name__ == "__main__":
    main()
