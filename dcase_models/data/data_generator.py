import os
import numpy as np
import inspect
import random

from keras.utils import Sequence

from .feature_extractor import FeatureExtractor
from .dataset_base import Dataset
from ..utils.ui import progressbar
from ..utils.data import get_fold_val
from ..utils.files import mkdir_if_not_exists
from ..utils.files import duplicate_folder_structure
from ..utils.files import list_wav_files


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

    def __init__(self, dataset, feature_extractor, **kwargs):
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

        feature_name = type(feature_extractor).__name__
        mkdir_if_not_exists(
            os.path.join(dataset.dataset_path, self.features_folder)
        )
        self.features_path = os.path.join(
            dataset.dataset_path, self.features_folder, feature_name
        )
        mkdir_if_not_exists(self.features_path)

        if not self.dataset.check_sampling_rate(feature_extractor.sr):
            print('Changing sampling rate ...')
            self.dataset.change_sampling_rate(feature_extractor.sr)
            print('Done!')

        self.audio_path = self.dataset.get_audio_paths(feature_extractor.sr)

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

    def load_data(self):
        """ Creates self.data that contains features and annotations for all files
        in all folds
        """

        if not self.check_if_features_extracted:
            self.extract_features()

        self.dataset.generate_file_lists()
        audio_path, subfolders = self.dataset.get_audio_paths(
            self.feature_extractor.sr
        )
        if subfolders is None:
            subfolders = ['']

        for fold in progressbar(self.dataset.fold_list, prefix='fold: '):
            self.data[fold] = {'X': [], 'Y': []}
            for subfolder in subfolders:
                subfolder_name = os.path.basename(subfolder)
                files_audio = self.dataset.file_lists[fold]
                file_features = self.convert_audio_path_to_features_path(
                    files_audio, subfolder=subfolder_name
                )
                X, Y = self.data_generation(file_features)
                self.data[fold]['X'].extend(X)
                self.data[fold]['Y'].extend(Y)

    def get_data_for_training(self, fold_test, upsampling=False):
        """ Returns arrays and lists for use in training process

        Parameters
        ----------
        fold_test : str
            Name of fold test
        upsampling : bool, optional
            If true, the training set is upsampled to get a balanced array

        Return
        ----------
        X_train : ndarray
            3D array with all instances of train set.
            Shape: (N_instances, N_hops, N_freqs)
        Y_train : list of ndarray
            2D array with annotations (one-hot coding)
            Shape: (N_instances, N_classes)
        X_val : list of ndarray
            List of features for each file of validation set
        Y_val : list of ndarray
            List of annotations matrix for each file in validation set

        """
        if self.evaluation_mode in ['cross-validation',
                                    'cross-validation-with-test']:
            # cross-validation mode
            fold_val = get_fold_val(fold_test, self.dataset.fold_list)
            folds_train = self.dataset.fold_list.copy()
            folds_train.remove(fold_test)
            if self.use_validate_set:
                folds_train.remove(fold_val)
                X_val = self.data[fold_val]['X']
                Y_val = self.data[fold_val]['Y']

            X_train_list = []
            Y_train_list = []
            for fold in folds_train:
                X_train_list.extend(self.data[fold]['X'])
                Y_train_list.extend(self.data[fold]['Y'])

            X_train = np.concatenate(X_train_list, axis=0)
            Y_train = np.concatenate(Y_train_list, axis=0)

            X_train_up = X_train.copy()
            Y_train_up = Y_train.copy()

            # upsampling
            if upsampling:
                n_classes = Y_train.shape[1]
                Ns = np.zeros(n_classes)
                for j in range(n_classes):
                    Ns[j] = np.sum(Y_train[:, j] == 1)
                Ns = np.floor(np.amax(Ns)/Ns)-1
                for j in range(n_classes):
                    if Ns[j] > 1:
                        X_j = X_train[Y_train[:, j] == 1]
                        Y_j = Y_train[Y_train[:, j] == 1]
                        X_train_up = np.concatenate(
                            [X_train_up]+[X_j]*int(Ns[j]), axis=0)
                        Y_train_up = np.concatenate(
                            [Y_train_up]+[Y_j]*int(Ns[j]), axis=0)

            if self.evaluation_mode == 'cross-validation-with-test':
                return (
                    X_train_up, Y_train_up,
                    self.data[fold_test]['X'].copy(),
                    self.data[fold_test]['Y'].copy()
                )

            if self.use_validate_set:
                return X_train_up, Y_train_up, X_val, Y_val
            else:
                return X_train_up, Y_train_up, X_train_list, Y_train_list

        if (self.evaluation_mode == 'train-validate-test'):
            # train-val-test mode
            X_val = self.data['validate']['X'].copy()
            Y_val = self.data['validate']['Y'].copy()

            X_train = np.concatenate(self.data['train']['X'], axis=0)
            Y_train = np.concatenate(self.data['train']['Y'], axis=0)

            return X_train, Y_train, X_val, Y_val

        if (self.evaluation_mode == 'train-test'):
            # train-test mode
            X_val = self.data['train']['X'].copy()
            Y_val = self.data['train']['Y'].copy()

            X_train = np.concatenate(self.data['train']['X'], axis=0)
            Y_train = np.concatenate(self.data['train']['Y'], axis=0)

            return X_train, Y_train, X_val, Y_val

    def get_data_for_testing(self, fold_test):
        """ Returns lists to use to evaluate a model

        Parameters
        ----------
        fold_test : str
            Name of fold test

        Return
        ----------
        X_test : list of ndarray
            List of features for each file of test set
        Y_test : list of ndarray
            List of annotations matrix for each file in test set

        """
        # cross-validation mode
        X_test = self.data[fold_test]['X'].copy()
        Y_test = self.data[fold_test]['Y'].copy()

        return X_test, Y_test

    def get_one_example_per_file(self, fold_test, train=True):
        # cross-validation mode

        if train:
            if self.evaluation_mode == 'cross-validation':
                fold_val = get_fold_val(fold_test, self.dataset.fold_list)
                folds_train = self.dataset.fold_list.copy()
                folds_train.remove(fold_test)
                folds_train.remove(fold_val)
            if self.evaluation_mode in ['train-test', 'train-validate-test']:
                folds_train = ['train']

            # X_val = self.data[fold_val]['X']
            # Y_val = self.data[fold_val]['Y']

            X_train = []
            Y_train = []
            Files_names_train = []
            for fold_train in folds_train:
                for file in range(len(self.data[fold_train]['X'])):
                    X = self.data[fold_train]['X'][file]
                    if len(X) == 0:
                        continue
                    ix = int(len(X)/2) if len(X) > 1 else 0
                    X = np.expand_dims(
                        self.data[fold_train]['X'][file][ix], axis=0)
                    X_train.append(X)
                    Y = np.expand_dims(
                        self.data[fold_train]['Y'][file][ix], axis=0)
                    Y_train.append(Y)
                    if self.dataset.file_lists is not None:
                        Files_names_train.append(
                            self.dataset.file_lists[fold_train][file]
                        )

            X_train = np.concatenate(X_train, axis=0)
            Y_train = np.concatenate(Y_train, axis=0)

            return X_train, Y_train, Files_names_train

        else:
            X_test = []
            Y_test = []
            Files_names_test = []

            for file in range(len(self.data[fold_test]['X'])):
                X = self.data[fold_test]['X'][file]
                if len(X) == 0:
                    continue
                ix = int(len(X)/2) if len(X) > 1 else 0
                X = np.expand_dims(
                    self.data[fold_test]['X'][file][ix], axis=0)
                X_test.append(X)
                Y = np.expand_dims(
                    self.data[fold_test]['Y'][file][ix], axis=0)
                Y_test.append(Y)
                if self.dataset.file_lists is not None:
                    Files_names_test.append(
                        self.dataset.file_lists[fold_test][file]
                    )
            X_test = np.concatenate(X_test, axis=0)
            Y_test = np.concatenate(Y_test, axis=0)

            return X_test, Y_test, Files_names_test

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


    def extract_features(self):
        """
        Extracts features of each wav file present in self.audio_path.

        If the dataset is nos sampled at  the sampling rate set in
        feature_extractor.sr, the dataset is resampled before
        feature extraction.

        """

        # Change sampling rate if needed
        if not self.dataset.check_sampling_rate(self.feature_extractor.sr):
            self.dataset.change_sampling_rate(self.feature_extractor.sr)

        # Define path to audio and features folders
        audio_path, subfolders = self.dataset.get_audio_paths(
            self.feature_extractor.sr
        )

        # Duplicate folder structure of audio in features folder
        duplicate_folder_structure(audio_path, self.features_path)

        for audio_folder in subfolders:
            subfolder_name = os.path.basename(audio_folder)
            features_path = os.path.join(self.features_path, subfolder_name)
            if not self.feature_extractor.check_if_extracted(features_path):
                # input()
                # Navigate in the structure of audio folder and extract
                # features of the each wav file
                for path_to_file_audio in list_wav_files(audio_folder):
                    features_array = self.feature_extractor.calculate(
                        path_to_file_audio
                    )
                    path_to_features_file = path_to_file_audio.replace(
                        audio_folder, features_path
                    )
                    path_to_features_file = path_to_features_file.replace(
                        'wav', 'npy'
                    )
                    np.save(path_to_features_file, features_array)

                # Save parameters.json for future checking
                self.feature_extractor.set_as_extracted(features_path)

    def check_if_features_extracted(self):
        """
        Check if the features were extracted before.

        """
        audio_path, subfolders = self.dataset.get_audio_paths(
            self.feature_extractor.sr
        )
        for audio_folder in subfolders:
            subfolder_name = os.path.basename(audio_folder)
            features_path = os.path.join(self.features_path, subfolder_name)
            feat_extracted = self.feature_extractor.check_if_extracted(
                features_path)
            if not feat_extracted:
                return False

        return True


class KerasDataGenerator(Sequence):

    def __init__(self, data_generator, folds,
                 batch_size=32, shuffle=True,
                 validation=False, scaler=None):
        self.data_gen = data_generator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation = validation
        self.folds = folds
        self.scaler = scaler

        self.file_list = []

        self.data_gen.dataset.generate_file_lists()
        audio_path, subfolders = self.data_gen.dataset.get_audio_paths(
            self.data_gen.feature_extractor.sr
        )

        for fold in folds:
            #self.file_list.extend(data_generator.file_lists[fold])
            for subfolder in subfolders:
                subfolder_name = os.path.basename(subfolder)
                files_audio = self.data_gen.dataset.file_lists[fold]
                file_features = self.data_gen.convert_audio_path_to_features_path(
                    files_audio, subfolder=subfolder_name
                )
                self.file_list.extend(file_features)

        self.on_epoch_end()

    def load_batch(self, index):
        list_file_batch = self.file_list[
            index*self.batch_size:(index+1)*self.batch_size
        ]

        # Generate data
        X, Y = self.data_gen.data_generation(list_file_batch)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        if not self.validation:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

        return X, Y   

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        return self.load_batch(index)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            random.shuffle(self.file_list)
