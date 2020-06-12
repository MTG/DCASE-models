import glob
import os
import numpy as np

from ..utils.ui import progressbar
from ..utils.data import get_fold_val
from ..utils.files import download_files_and_unzip
# from ..utils.files import mkdir_if_not_exists


class DataGenerator():
    """
    DataGenerator includes functions to load and manage DCASE datasets.
    This class can be redefined by child classes (see UrbanSound8k, ESC50)

    Attributes
    ----------
    audio_folder : str
        Path to the folder with audio files
    features_folder : str
        Path to the folder with the features files
    annotations_folder : str
        Path to the folder with the annotation files
    features : list of str
        Names of features to be loaded
    fold_list : list of str
        List of fold names
    label_list : list of str
        List of label names
    meta_file : str, optional
        Path to the metadata file of dataset, default None
    file_lists : dict
        Dict of form {'fold1' : [file1, file2 ...], ...}
    data : dict
        Dict that contains the data.
        Form: {'fold1' : {'X' : [features1, features2, ...],
                          'Y': [ann1, ann2, ...]}, ...}

    Methods
    -------
    get_file_lists()
        Create self.file_lists, a dict thath includes a list of files per fold
    get_annotations(file_name, features):
        Returns the annotations of file in file_name path
    data_generation(list_files_temp):
        Returns features and annotations for all files in list_files_temp
    load_data():
        Creates self.data that contains features and annotations for all files
        in all folds
    get_data_for_training(fold_test, upsampling=False):
        Returns arrays and lists for use in training process
    get_data_for_testing(fold_test):
        Returns lists to use to evaluate a model
    get_one_example_per_file(fold_test)
        Similar to get_data_for_training, but returns only one
        example (sequence) for each file
    return_file_list()
        Returns self.file_lists
    """

    def __init__(self, dataset_path, features_folder, features,
                 audio_folder=None, use_validate_set=True):
        """ Initialize the DataGenerator
        Parameters
        ----------
        audio_folder : str
            Path to the folder with audio files
        features_folder : str
            Path to the folder with the features files
        annotations_folder : str
            Path to the folder with the annotation files
        features : list of str
            Names of features to be loaded
        fold_list : list of str
            List of fold names
        label_list : list of str
            List of label names
        meta_file : str, optional
            Path to the metadata file of dataset, default None

        """
        # General attributes
        self.dataset_path = dataset_path
        self.features = features
        self.use_validate_set = use_validate_set

        if audio_folder is None:
            self.audio_folder = os.path.join(dataset_path, 'audio')
        else:
            self.audio_folder = os.path.join(dataset_path, audio_folder)
        self.features_folder = os.path.join(dataset_path, features_folder)

        # Specific attributes
        self.build()

        # check if the dataset was download
        # TODO improve this
        # if not self.check_if_dataset_was_downloaded():
        #    response = input('The dataset was not downloaded : download [y]
        #                       or continue without downloading [n] : ')
        #    if response == 'y':
        #        self.download_dataset()

        # make dataset folder if does not exists
        # mkdir_if_not_exists(self.dataset_path)

        # make features folder if does not exists
        # if self.check_if_dataset_was_downloaded():
        #    mkdir_if_not_exists(self.features_folder)

        self.file_lists = {}
        self.data = {}
        self.get_file_lists()

    def build(self):
        self.fold_list = ["fold1", "fold2", "fold3", "fold4",
                          "fold5", "fold6", "fold7", "fold8",
                          "fold9", "fold10"]
        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]
        # self.meta_file = None
        # self.taxonomy_file = None
        self.evaluation_mode = 'cross-validation'

        self.folders_list = []
        for fold in self.fold_list:
            audio_path = os.path.join(self.audio_folder, fold)
            features_path = os.path.join(self.features_folder, fold)
            audio_features_path = {'audio': audio_path,
                                   'features': features_path}
            self.folders_list.append(audio_features_path)

    def get_file_lists(self):
        """ Create self.file_lists, a dict thath includes a list of files per fold
        """
        for fold in self.fold_list:
            features_folder = os.path.join(
                self.features_folder, fold, self.features)
            self.file_lists[fold] = sorted(
                glob.glob(os.path.join(features_folder, '*.npy')))

    def get_annotations(self, file_name, features):
        """ Returns the annotations of file in file_name path
        Parameters
        ----------
        file_name : str
            Path to the file
        features : ndarray
            3D array with the features of file_name

        Returns
        -------
        ndarray
            annotations of the file file_name

        """
        y = np.zeros((len(features), len(self.label_list)))
        class_ix = int(os.path.basename(file_name).split('-')[1])
        y[:, class_ix] = 1
        return y

    def data_generation(self, list_files_temp):
        """ Returns features and annotations for all files in list_files_temp

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

        for file_name in list_files_temp:
            features = np.load(file_name)
            features_list.append(features)
            y = self.get_annotations(file_name, features)
            annotations.append(y)

        return features_list, annotations

    def load_data(self):
        """ Creates self.data that contains features and annotations for all files
        in all folds
        """
        for fold in progressbar(self.fold_list, prefix='fold: '):
            X, Y = self.data_generation(self.file_lists[fold])
            self.data[fold] = {'X': X, 'Y': Y}

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
        # cross-validation mode
        fold_val = get_fold_val(fold_test, self.fold_list)
        folds_train = self.fold_list.copy()  # list(range(1,N_folds+1))
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

        if self.use_validate_set:
            return X_train_up, Y_train_up, X_val, Y_val
        else:
            return X_train_up, Y_train_up, X_train_list, Y_train_list

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
        X_test = self.data[fold_test]['X']
        Y_test = self.data[fold_test]['Y']

        return X_test, Y_test

    def get_one_example_per_file(self, fold_test):
        # cross-validation mode
        fold_val = get_fold_val(fold_test, self.fold_list)
        folds_train = self.fold_list.copy()
        folds_train.remove(fold_test)
        folds_train.remove(fold_val)

        # X_val = self.data[fold_val]['X']
        # Y_val = self.data[fold_val]['Y']

        X_train = []
        Y_train = []
        Files_names_train = []
        for fold_train in folds_train:
            for file in range(len(self.data[fold_train]['X'])):
                X = self.data[fold_train]['X'][file]
                if len(X) <= 1:
                    continue
                ix = int(len(X)/2)
                X = np.expand_dims(
                    self.data[fold_train]['X'][file][ix], axis=0)
                X_train.append(X)
                Y = np.expand_dims(
                    self.data[fold_train]['Y'][file][ix], axis=0)
                Y_train.append(Y)
                if self.file_lists is not None:
                    Files_names_train.append(self.file_lists[fold_train][file])
        X_train = np.concatenate(X_train, axis=0)
        Y_train = np.concatenate(Y_train, axis=0)

        return X_train, Y_train, Files_names_train

    def return_file_list(self, fold_test):
        return self.file_lists[fold_test]

    def get_folder_lists(self):
        return self.folders_list

    def download_dataset(self, zenodo_url, zenodo_files):
        if self.check_if_dataset_was_downloaded():
            response = input(
                'The dataset was downloaded already: download again [y]' +
                ' or continue [n] : ')
            if response == 'n':
                return None
        download_files_and_unzip(self.dataset_path, zenodo_url, zenodo_files)

    def set_dataset_download_finish(self):
        log_file = os.path.join(self.dataset_path, 'download.txt')
        with open(log_file, 'w') as txt_file:
            txt_file.write('The dataset was download ...\n')

    def check_if_dataset_was_downloaded(self):
        log_file = os.path.join(self.dataset_path, 'download.txt')
        return os.path.exists(log_file)

    def convert_features_path_to_audio_path(self, fetures_path):
        ''' convert ../features/foldX/MelSpectrogram/x.npy
            to ../audio/foldX/x.wav '''
        audio_folder = self.audio_folder.split('/')[-1]
        features_folder = self.features_folder.split('/')[-1]
        audio_path = fetures_path.replace(self.features+'/', '')
        audio_path = audio_path.replace(features_folder, audio_folder)
        audio_path = audio_path.replace('.npy', '.wav')
        return audio_path
