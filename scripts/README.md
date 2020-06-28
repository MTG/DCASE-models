<pre>
  ____   ____    _    ____  _____                          _      _     
 |  _ \ / ___|  / \  / ___|| ____|     _ __ ___   ___   __| | ___| |___ 
 | | | | |     / _ \ \___ \|  _| _____| '_ ` _ \ / _ \ / _` |/ _ \ / __|
 | |_| | |___ / ___ \ ___) | |__|_____| | | | | | (_) | (_| |  __/ \__ \
 |____/ \____/_/   \_\____/|_____|    |_| |_| |_|\___/ \__,_|\___|_|___/
                                                                       
</pre>

# Scripts

This folder contains Python scripts to demonstrate `DCASE-models`. It includes 6 scripts that implement functionalities for the different parts of a DCASE related system.

- [Dataset downloading](download_dataset.py)
- [Data augmentation](data_augmentation.py)
- [Feature extraction](feature_extraction.py)
- [Model training](train_model.py)
- [Model evaluation](evaluate_model.py)
- [Fine tuning](fine_tuning.py)

## Usage

First, note that the default parameters are stored in [`parameters.json`](`parameters.json`) file. You can use other `parameters.json` by passing its path in -p (or --path) argument of each script.

In the next, we show examples on how to use these scripts for the complete development pipeline. For further usage information you can access to each script instructions by:
```
python download_dataset.py --help
```

### Download a dataset
First let's start by downloading a dataset. For instance to download ESC-50 dataset:
```
python download_dataset.py -d ESC50
```

> Note that the dataset will be downloaded in `../datasets/ESC50` following the path set in [`parameters.json`](`../parameters.json`). 

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

> See that you have to pass the features name as an argument. Available features representations are in [features.py](../dcase_models/data/features.py).

### Model training
To train the model is also very easy. For instance, to train `SB_CNN` model on ESC-50 dataset with the `MelSpectrogram` features extracted before:
```
python train_model.py -d ESC50 -f MelSpectrogram -m SB_CNN -fold fold1
```

> In this case, you have to pass the model name and a fold name. This is considered to be the fold for testing, meaning that this fold will not be used during training.

### Model evaluation
Once the model is trained, you can evaluate the model in the test set:
```
python evaluate_model.py -d ESC50 -f MelSpectrogram -m SB_CNN -fold fold1
```

This scripts prints the results that we get from `[sed_eval](https://tut-arg.github.io/sed_eval/)` library.

### Fine-tuning
Once you have a model trained in some dataset, you can fine-tune this model on other dataset. For instance to fine-tune the model trained before on MAVD dataset just:
```
python fine_tuning.py -od ESC50 -ofold fold1 -f MelSpectrogram -m SB_CNN -d MAVD -fold test
```

> Note that the information of the origin dataset is passed in -od and -ofold arguments. Besides -d and -fold are the destination dataset and the fold test respectively.
