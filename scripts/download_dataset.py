'''
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
from dcase_models.utils.files import load_json



def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, help='dataset name (e.g. UrbanSound8k,'\
                        ' ESC50, ESC10, URBAN_SED, SONYC_UST)',
                        default='UrbanSound8k')
    parser.add_argument('-p', '--path', type=str, help='path to the parameters.json file',
                        default='../')
    args = parser.parse_args()

    if args.dataset not in get_available_datasets():
        raise AttributeError('Dataset not available')

    # Get parameters
    parameters_file = os.path.join(args.path, 'parameters.json')
    params = load_json(parameters_file)
    params_dataset = params['datasets'][args.dataset]

    # Get and init dataset class
    dataset_class = get_available_datasets()[args.dataset]
    dataset_path = os.path.join(args.path, params_dataset['dataset_path'])
    dataset = dataset_class(dataset_path)

    # Download dataset
    if dataset.check_if_dataset_was_downloaded():
        resp = input(
            '''%s dataset is already there. It has been downloaded before.
            Do you want to download it anyway? [n] : ''' % args.dataset
        )
        if resp == 'y':
            dataset.download_dataset(force_download=True)
    else:
        dataset.download_dataset()
    print('Done!')


if __name__ == "__main__":
    main()
