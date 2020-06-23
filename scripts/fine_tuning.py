r'''
  ____   ____    _    ____  _____                          _      _
 |  _ \ / ___|  / \  / ___|| ____|     _ __ ___   ___   __| | ___| |___
 | | | | |     / _ \ \___ \|  _| _____| '_ ` _ \ / _ \ / _` |/ _ \ / __|
 | |_| | |___ / ___ \ ___) | |__|_____| | | | | | (_) | (_| |  __/ \__ \\
 |____/ \____/_/   \_\____/|_____|    |_| |_| |_|\___/ \__,_|\___|_|___/

 Model fine-tuning example

'''

import os
import argparse

from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.model.models import get_available_models
from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.scaler import Scaler
from dcase_models.util.files import load_json
from dcase_models.util.files import mkdir_if_not_exists, save_pickle
from dcase_models.util.data import evaluation_setup

sed_datasets = ['URBAN_SED', 'TUTSoundEvents2017', 'MAVD']


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-od', '--origin_dataset', type=str,
        help='dataset name (e.g. UrbanSound8k, ESC50, URBAN_SED, SONYC_UST)',
        default='UrbanSound8k'
    )
    parser.add_argument(
        '-ofold', '--origin_fold_name', type=str,
        help='origin fold name',
        default='fold1')
    parser.add_argument(
        '-d', '--dataset', type=str,
        help='dataset name (e.g. UrbanSound8k, ESC50, URBAN_SED, SONYC_UST)',
        default='ESC50'
    )
    parser.add_argument(
        '-fold', '--fold_name', type=str,
        help='destination fold name',
        default='fold1')
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
    parser.add_argument(
        '-m', '--model', type=str,
        help='model name (e.g. MLP, SB_CNN, SB_CNN_SED, A_CRNN, VGGish)',
        default='SB_CNN')

    parser.add_argument(
        '-s', '--models_path', type=str,
        help='path to save the trained model',
        default='../trained_models'
    )
    args = parser.parse_args()

    print(__doc__)

    if args.dataset not in get_available_datasets():
        raise AttributeError('Dataset not available')

    if args.features not in get_available_features():
        raise AttributeError('Features not available')

    if args.model not in get_available_models():
        raise AttributeError('Model not available')

    # Get parameters
    parameters_file = os.path.join(args.path, 'parameters.json')
    params = load_json(parameters_file)
    params_dataset = params['datasets'][args.dataset]
    params_features = params['features']
    params_model = params['models'][args.model]

    # Load origin model
    model_path_origin = os.path.join(args.models_path, args.model,
                                     args.origin_dataset)
    model_class = get_available_models()[args.model]
    metrics = ['accuracy']
    if args.dataset in sed_datasets:
        metrics = ['sed']
    model_container = model_class(
        model=None, model_path=model_path_origin,
        metrics=metrics
    )
    model_container.load_model_weights(
        os.path.join(model_path_origin, args.origin_fold_name))

    kwargs = {}
    if args.dataset in sed_datasets:
        kwargs = {'sequence_hop_time': params_features['sequence_hop_time']}

    # Get and init dataset class
    dataset_class = get_available_datasets()[args.dataset]
    dataset_path = os.path.join(args.path, params_dataset['dataset_path'])
    dataset = dataset_class(dataset_path, **kwargs)

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
    print('Features shape: ', features.get_shape())

    # Check if features were extracted
    if not features.check_if_extracted(dataset):
        print('Extracting features ...')
        features.extract(dataset)
        print('Done!')

    use_validate_set = True
    if args.dataset in ['TUTSoundEvents2017', 'ESC50', 'ESC10']:
        # When have less data, don't use validation set.
        use_validate_set = False

    folds_train, folds_val, _ = evaluation_setup(
        args.fold_name, dataset.fold_list,
        params_dataset['evaluation_mode'],
        use_validate_set=use_validate_set
    )

    data_gen_train = DataGenerator(
        dataset, features, folds=folds_train,
        batch_size=params['train']['batch_size'],
        shuffle=True, train=True, scaler=None
    )

    scaler = Scaler(normalizer=params_model['normalizer'])
    print('Fitting features ...')
    scaler.fit(data_gen_train)
    print('Done!')

    data_gen_train.set_scaler(scaler)

    data_gen_val = DataGenerator(
        dataset, features, folds=folds_val,
        batch_size=params['train']['batch_size'],
        shuffle=False, train=False, scaler=scaler
    )

    # Fine-tune model
    n_classes = len(dataset.label_list)
    layer_where_to_cut = -2
    model_container.fine_tuning(layer_where_to_cut,
                                new_number_of_classes=n_classes,
                                new_activation='sigmoid',
                                freeze_source_model=True)

    model_container.model.summary()

    # Set paths
    model_folder = os.path.join(
        args.models_path, args.model,
        args.origin_dataset+'_ft_'+args.dataset)
    exp_folder = os.path.join(model_folder, args.fold_name)
    mkdir_if_not_exists(exp_folder, parents=True)

    # Save model json and scaler
    model_container.save_model_json(model_folder)
    save_pickle(scaler, os.path.join(exp_folder, 'scaler.pickle'))

    # Train model
    model_container.train(
        data_gen_train, data_gen_val,
        label_list=dataset.label_list,
        weights_path=exp_folder,
        sequence_time_sec=params_features['sequence_hop_time'],
        **params['train'])


if __name__ == "__main__":
    main()
