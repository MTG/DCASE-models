from dcase_models.data.datasets import get_available_datasets
from dcase_models.utils.files import load_json

import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name',
                        default='UrbanSound8k')
    args = parser.parse_args()

    if args.dataset not in get_available_datasets():
        raise AttributeError('Dataset not available')

    # Get parameters
    params = load_json('parameters.json')
    params_dataset = params['datasets'][args.dataset]

    # Get and init dataset class
    dataset_class = get_available_datasets()[args.dataset]
    dataset = dataset_class(params_dataset['dataset_path'])

    # Download dataset
    if dataset.check_if_dataset_was_downloaded():
        resp = input(
            '''%s was downloaded already,
            download it again? [n] : ''' % args.dataset
        )
        if resp == 'y':
            dataset.download_dataset(force_download=True)
    else:
        dataset.download_dataset()
    print('Done!')


if __name__ == "__main__":
    main()
