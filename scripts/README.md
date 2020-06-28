<pre>
  ____   ____    _    ____  _____                          _      _     
 |  _ \ / ___|  / \  / ___|| ____|     _ __ ___   ___   __| | ___| |___ 
 | | | | |     / _ \ \___ \|  _| _____| '_ ` _ \ / _ \ / _` |/ _ \ / __|
 | |_| | |___ / ___ \ ___) | |__|_____| | | | | | (_) | (_| |  __/ \__ \
 |____/ \____/_/   \_\____/|_____|    |_| |_| |_|\___/ \__,_|\___|_|___/
                                                                       
</pre>

# Scripts

This folder contains Python scripts to demonstrate `DCASE-models`. It includes 6 scripts that implement different functionalities around the pipeline of a DCASE related system.

- [Dataset downloading](scripts/download_dataset.py)
- [Data augmentation](scripts/data_augmentation.py)
- [Feature extraction](scripts/feature_extraction.py)
- [Model training](scripts/train_model.py)
- [Model evaluation](scripts/evaluate_model.py)
- [Fine tuning](scripts/fine_tuning.py)

## Usage

First, note that the default parameters are stored in [`parameters.json`](`parameters.json`) file.

In the next, we show an example of the complete development pipeline using these scripts. For further usage information of each script you can access to its instructions by:
```
python download_dataset.py --help
```

### Download a dataset
First let's start by downloading a dataset. For instance to download ESC-50 dataset:
```
python download_dataset.py -d ESC50
```

> Note that the dataset will be downloaded in `DCASE-models/datasets/ESC50` following the path set in [`parameters.json`](`parameters.json`). You can use other `parameters.json` by passing its path in -p/--path argument.

### Data augmentation
If you wan to use data augmentation techniques, you can run the following script.
```
python data_augmentation.py -d ESC50
```

### Feature extraction
Now, you can extract the features for each file in the dataset by:
```
python extract_features.py -d ESC50 -f MelSpectrogram
```

> See that you have to pass the features name as an argument. Available features representations are in [features.py](dcase_models/data/features.py).

### Model training
To train the model is also very easy. For instance, to train `SB_CNN` model on ESC-50 dataset with the `MelSpectrogram` features extracted before:
```
python train_model.py -d ESC50 -f MelSpectrogram -m SB_CNN -fold fold1
```

> In this case, you have to pass the model name and a fold name. This is considered to be the fold for testing, meaning that this fold will not be used during training.

### Model evaluation
Once the model is trained, you can evaluate the results in the test set:
```
python evaluate_model.py -d ESC50 -f MelSpectrogram -m SB_CNN -fold fold1
```

This scripts prints the results that we get from sed_eval library.

Datasets should be stored on [`/datasets`](https://github.com/pzinemanas/DCASE-models/tree/master/datasets).

Notebooks are designed to be self-contained, but datasets can be downloaded beforehand  as shown on [`/basics/download_and_prepare_datasets.ipynb`](https://github.com/pzinemanas/DCASE-models/blob/master/notebooks/basics/download_and_prepare_datasets.ipynb).

Basics notebooks can be run sequentially as a tutorial.

Default parameters for each model/dataset are stored in [`parameters.json`](https://github.com/pzinemanas/DCASE-models/blob/master/parameters.json) on the root directory. Notebooks provide examples on how to modify these parameters.
