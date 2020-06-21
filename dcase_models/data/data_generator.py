import os
import numpy as np
import inspect
import random

from keras.utils import Sequence

from .feature_extractor import FeatureExtractor
from .dataset_base import Dataset


class DataGenerator():
    """
    DataGenerator includes methods to load features files from DCASE datasets.

    It's also used to calculate features in each file in the dataset.

    Attributes
    ----------
    dataset : Dataset or classes inherited from.
        Instance of Dataset.
    feature_extractor : FeatureExtractor or classes inherited from.
        Instance of FeatureExtractor. This is only needed to use
        functions related to feature extraction
    features_folder : str
        Name of to the folder with the features files,
        default is 'features'.
    use_validate_set : bool
        If True, use validate set.
    features_path : str
        Path to features folder.
    audio_path : str
        Path to the audio folder.

    Methods
    -------
    data_generation(list_files):
        Returns features and annotations for all files in list_files.
    load_data():
        Creates self.data that contains features and annotations for all files
        in all folds.
    get_data_for_training(fold_test, upsampling=False,
                          evaluation_mode='cross-validation'):
        Returns arrays and lists for use in training process
    get_data_for_testing(fold_test):
        Returns lists to use to evaluate a model
    get_one_example_per_file(fold_test)
        Similar to get_data_for_training, but returns only one
        example (sequence) for each file
    convert_features_path_to_audio_path(features_file)
        Convert the path to a feature file (or list of files)
        to the path to the audio file.
    convert_audio_path_to_features_path(audio_file)
        Convert the path to an audio file (or list of files)
        to the path to the feature file.
    extract_features()
        Calculate features for each file in the dataset.
    check_if_features_extracted()
        Check if the features were extracted before.
    """

    def __init__(self, dataset, feature_extractor, folds,
                 batch_size=32, shuffle=True,
                 train=True, scaler=None, **kwargs):
        """ Initialize the DataGenerator

        Parameters
        ----------
        dataset_path : str
            Path to the dataset fold
        feature_extractor : FeatureExtractor or classes inherited from.
            Instance of FeatureExtractor. This is only needed to use
            functions related to feature extraction

        """
        # General attributes
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.folds = folds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.scaler = scaler
        self.features_folder = kwargs.get('features_folder', 'features')
        self.use_validate_set = kwargs.get('use_validate_set', True)
        self.evaluation_mode = kwargs.get(
            'evaluation_mode', 'cross-validation'
        )

        if (Dataset not in inspect.getmro(dataset.__class__)):
            raise AttributeError(
                'dataset has to be an instance of Dataset or similar'
            )

        if FeatureExtractor not in inspect.getmro(feature_extractor.__class__):
            raise AttributeError('''feature_extractor has to be an
                                    instance of FeatureExtractor or similar''')

        self.features_path = self.feature_extractor.get_features_path(
            dataset
        )

        self.audio_path = self.dataset.get_audio_paths(feature_extractor.sr)

        self.features_file_list = []
        self.dataset.generate_file_lists()
        audio_path, subfolders = self.dataset.get_audio_paths(
            self.feature_extractor.sr
        )
        for fold in folds:
            for subfolder in subfolders:
                subfolder_name = os.path.basename(subfolder)
                files_audio = self.dataset.file_lists[fold]
                file_features = self.convert_audio_path_to_features_path(
                    files_audio, subfolder=subfolder_name
                )
                self.features_file_list.extend(file_features)

        if shuffle:
            self.shuffle_list()

        self.data = {}

    def data_generation(self, list_files):
        """ Returns features and annotations for all files in list_files

        Parameters
        ----------
        list_IDs_temp : list
            List of file names

        Return
        ----------
        features_list : list of ndarray
            List of features for each file
        annotations : list of ndarray
            List of annotations matrix for each file

        """
        features_list = []
        annotations = []

        for file_name in list_files:
            features = np.load(file_name)
            features_list.append(features)
            file_audio = self.convert_features_path_to_audio_path(file_name)
            file_audio = self.paths_remove_aug_subfolder(file_audio)
            y = self.dataset.get_annotations(file_audio, features)
            annotations.append(y)

        return features_list, annotations

    def get_data(self):
        X_list, Y_list = self.data_generation(self.features_file_list)

        if self.scaler is not None:
            X_list = self.scaler.transform(X_list)

        if self.train:
            X = np.concatenate(X_list, axis=0)
            Y = np.concatenate(Y_list, axis=0)
        else:
            X = X_list.copy()
            Y = Y_list.copy()

        return X, Y

    def get_data_batch(self, index):
        list_file_batch = self.features_file_list[
            index*self.batch_size:(index+1)*self.batch_size
        ]

        # Generate data
        X_list, Y_list = self.data_generation(list_file_batch)

        if self.scaler is not None:
            X_list = self.scaler.transform(X_list)

        if self.train:
            X = np.concatenate(X_list, axis=0)
            Y = np.concatenate(Y_list, axis=0)
        else:
            X = X_list.copy()
            Y = Y_list.copy()

        return X, Y

    def convert_features_path_to_audio_path(self, features_file):
        """
        Convert the path to a feature file (or list of files)
        to the path to the audio file.

        Parameters
        ----------
        features_file : str or list of str
            Path to the features file or files.

        Return
        ----------
        audio_file : str or list of str
            Path to the audio file or files.

        """

        if type(features_file) is str:
            audio_file = features_file.replace(
                self.features_path, self.dataset.audio_path
            )
            audio_file = audio_file.replace('.npy', '.wav')
        elif type(features_file) is list:
            audio_file = []
            for j in range(len(features_file)):
                audio_file_j = features_file[j].replace(
                    self.features_path, self.dataset.audio_path
                )
                audio_file_j = audio_file_j.replace('.npy', '.wav')
                audio_file.append(audio_file_j)
        return audio_file

    def convert_audio_path_to_features_path(self, audio_file, subfolder=''):
        """
        Convert the path to an audio file (or list of files)
        to the path to the feature file.

        Parameters
        ----------
        audio_file : str or list of str
            Path to the audio file or files.

        Return
        ----------
        audio_file : str or list of str
            Path to the features file or files.

        """
        features_path = os.path.join(self.features_path, subfolder)
        if type(audio_file) is str:
            features_file = audio_file.replace(
                self.dataset.audio_path, features_path
            )
            features_file = features_file.replace('.wav', '.npy')
        elif type(audio_file) is list:
            features_file = []
            for j in range(len(audio_file)):
                features_file_j = audio_file[j].replace(
                    self.dataset.audio_path, features_path
                )
                features_file_j = features_file_j.replace('.wav', '.npy')
                features_file.append(features_file_j)

        return features_file

    def paths_remove_aug_subfolder(self, path):
        audio_path, subfolders = self.dataset.get_audio_paths()
        new_path = None
        for subfolder in subfolders:
            if subfolder in path:
                new_path = path.replace(subfolder, audio_path)
                break

        return new_path

    def shuffle_list(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            random.shuffle(self.features_file_list)

    def __len__(self):
        return int(np.ceil(len(self.features_file_list) / self.batch_size))

    def set_scaler(self, scaler):
        self.scaler = scaler


class KerasDataGenerator(Sequence):

    def __init__(self, data_generator):
        self.data_gen = data_generator
        self.data_gen.shuffle_list()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.data_gen)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        return self.data_gen.get_data_batch(index)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.data_gen.shuffle_list()
