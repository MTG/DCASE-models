import sys
import os
import numpy as np
import argparse

sys.path.append('../')
from dcase_models.utils.files import load_json
from dcase_models.data.datasets import get_available_datasets

parser = argparse.ArgumentParser(description='Test Data downloading')
parser.add_argument('-d', '--dataset', type=str, help='dataset to use for the test', default='UrbanSound8k')
args = parser.parse_args()

params = load_json('parameters.json')
params_dataset = params["datasets"][args.dataset]

# get dataset class
dataset_class = get_available_datasets()[args.dataset]
print(dataset_class)

dataset = dataset_class(params_dataset['dataset_path'])

dataset.download_dataset()