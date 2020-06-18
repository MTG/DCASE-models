'''
  ____   ____    _    ____  _____                          _      _
 |  _ \ / ___|  / \  / ___|| ____|     _ __ ___   ___   __| | ___| |___
 | | | | |     / _ \ \___ \|  _| _____| '_ ` _ \ / _ \ / _` |/ _ \ / __|
 | |_| | |___ / ___ \ ___) | |__|_____| | | | | | (_) | (_| |  __/ \__ \\
 |____/ \____/_/   \_\____/|_____|    |_| |_| |_|\___/ \__,_|\___|_|___/

 Evaluating model example

'''

import os
import argparse

from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.model.models import get_available_models
from dcase_models.data.data_generator import DataGenerator
from dcase_models.utils.files import load_json, load_pickle


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, help='dataset name (e.g. UrbanSound8k,'\
                        ' ESC50, ESC10, URBAN_SED, SONYC_UST)',
                        default='UrbanSound8k')
    parser.add_argument('-f', '--features', type=str, help='features name (e.g. Spectrogram,'\
                        ' MelSpectrogram, Openl3)',
                        default='MelSpectrogram')
    parser.add_argument('-p', '--path', type=str, help='path to the parameters.json file',
                        default='../')
    parser.add_argument('-m', '--model', type=str, help='model name (e.g. MLP, SB_CNN,'\
                        ' SB_CNN_SED, A_CRNN, VGGish)',
                        default='SB_CNN')
    parser.add_argument('-fold', '--fold_name', type=str, help='fold name',
                        default='fold1')
    parser.add_argument('-s', '--models_path', type=str, help='path to load the trained model',
                        default='../trained_models')
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

    # Get and init dataset class
    kwargs = {}
    if args.dataset in ['URBAN_SED', 'TUTSoundEvents2017']:
        kwargs = {'sequence_hop_time': params_features['sequence_hop_time']}
    dataset_class = get_available_datasets()[args.dataset]
    dataset = dataset_class(params_dataset['dataset_path'], **kwargs)

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
    if args.dataset == 'TUTSoundEvents2017':
        X_test, Y_test = data_generator.get_data_for_testing(args.fold_name)
    else:
        X_test, Y_test = data_generator.get_data_for_testing('test')

    # Set paths
    model_folder = os.path.join(args.models_path, args.model, args.dataset)
    exp_folder = os.path.join(model_folder, args.fold_name)

    # Load scaler
    scaler_file = os.path.join(exp_folder, 'scaler.pickle')
    scaler = load_pickle(scaler_file)

    X_test = scaler.transform(X_test)

    # Load model and best weights
    model_class = get_available_models()[args.model]
    metrics = ['accuracy']
    if args.dataset in ['URBAN_SED', 'TUTSoundEvents2017']:
        metrics = ['sed']
    model_container = model_class(model=None, model_path=model_folder, metrics=metrics)
    model_container.load_model_weights(exp_folder)

    kwargs = {}
    if args.dataset in ['URBAN_SED', 'TUTSoundEvents2017']:
        kwargs = {'sequence_time_sec':params_features['sequence_hop_time'],
                  'metric_resolution_sec':1.0, 'label_list': dataset.label_list}
    results = model_container.evaluate(X_test, Y_test, **kwargs)

    for metric in metrics:
        if metric == 'sed':
            print(results['sed'])
        else:
            print('%s in %s is %f' % (metric, args.fold_name, results[metric]))


if __name__ == "__main__":
    main()
