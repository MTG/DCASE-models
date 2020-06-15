import glob
import os
import numpy as np
import sox

from .feature_extractor import FeatureExtractor
from ..utils.ui import progressbar
from ..utils.data import get_fold_val
from ..utils.files import download_files_and_unzip
from ..utils.files import mkdir_if_not_exists
from ..utils.files import duplicate_folder_structure
from ..utils.files import list_wav_files_in_folder


class DataGenerator():
    """
    DataGenerator inlcudes methods to load features files from DCASE datasets.

    It's also used to calculate features in each file in the dataset.
    
    Attributes
    ----------
    dataset : Dataset or childs
        Instance of Dataset.
    feature_extractor : FeatureExtractor or childs
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
    get_data_for_training(fold_test, upsampling=False, evaluation_mode='cross-validation'):
        Returns arrays and lists for use in training process
    get_data_for_testing(fold_test):
        Returns lists to use to evaluate a model
    get_one_example_per_file(fold_test)
        Similar to get_data_for_training, but returns only one
        example (sequence) for each file
    convert_features_path_to_audio_path(features_file)
        Convert the path to a feature file (or list of files) to the path to the audio file.
    convert_audio_path_to_features_path(audio_file)
        Convert the path to an audio file (or list of files) to the path to the feature file.
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
        feature_extractor : FeatureExtractor or childs
            Instance of FeatureExtractor. This is only needed to use
            functions related to feature extraction

        """
        # General attributes
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.features_folder = kwargs.get('features_folder', 'features')
        self.use_validate_set = kwargs.get('use_validate_set', True)

        if (feature_extractor.__class__.__bases__[0] is not FeatureExtractor) and \
            (feature_extractor.__class__ is not FeatureExtractor):
            raise AttributeError('feature_extractor has to be an instance of FeatureExtractor')

        feature_name = type(feature_extractor).__name__
        mkdir_if_not_exists(os.path.join(dataset.dataset_path, self.features_folder))
        self.features_path = os.path.join(dataset.dataset_path, self.features_folder, feature_name)  
        mkdir_if_not_exists(self.features_path)

        if not self.dataset.check_sampling_rate(feature_extractor.sr):
            print('Changing sampling rate ...')
            self.dataset.change_sampling_rate(feature_extractor.sr)
            print('Done!')

        self.audio_path = self.dataset.get_audio_path(feature_extractor.sr)

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

        for fold in progressbar(self.dataset.fold_list, prefix='fold: '):
            file_audio = self.dataset.file_lists[fold]
            file_features = self.convert_audio_path_to_features_path(file_audio)
            X, Y = self.data_generation(file_features)
            self.data[fold] = {'X': X, 'Y': Y}

    def get_data_for_training(self, fold_test, upsampling=False, evaluation_mode='cross-validation'):
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
        if evaluation_mode == 'cross-validation':
            # cross-validation mode
            fold_val = get_fold_val(fold_test, self.dataset.fold_list)
            folds_train = self.dataset.fold_list.copy()  # list(range(1,N_folds+1))
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

        if (evaluation_mode == 'train-validate-test'):
            # train-val-test mode
            X_val = self.data['validate']['X']
            Y_val = self.data['validate']['Y'] #[1]  # grid time of metrics

            X_train = np.concatenate(self.data['train']['X'], axis=0)
            # grid time of instances for training
            Y_train = np.concatenate(self.data['train']['Y'], axis=0)  #[0]

            return  X_train, Y_train, X_val, Y_val    

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
            fold_val = get_fold_val(fold_test, self.dataset.fold_list)
            folds_train = self.dataset.fold_list.copy()
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
                    if len(X) == 0:
                        continue
                    #ix = int(len(X)/2)
                    ix = int(len(X)/2) if len(X) > 1 else 0
                    X = np.expand_dims(
                        self.data[fold_train]['X'][file][ix], axis=0)
                    X_train.append(X)
                    Y = np.expand_dims(
                        self.data[fold_train]['Y'][file][ix], axis=0)
                    Y_train.append(Y)
                    if self.dataset.file_lists is not None:
                        Files_names_train.append(self.dataset.file_lists[fold_train][file])
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
                    Files_names_test.append(self.dataset.file_lists[fold_test][file])
            X_test = np.concatenate(X_test, axis=0)
            Y_test = np.concatenate(Y_test, axis=0)

            return X_test, Y_test, Files_names_test
        

    def convert_features_path_to_audio_path(self, features_file):
        """ 
        Convert the path to a feature file (or list of files) to the path to the audio file.

        Parameters
        ----------
        features_file : str or list of str
            Path to the features file or files.

        Return
        ----------
        audio_file : str or list of str
            Path to the audio file or files.

        """

        #print(features_file)
        if type(features_file) is str:
            audio_file = features_file.replace(self.features_path, self.dataset.audio_path)
            audio_file = audio_file.replace('.npy', '.wav')
        elif type(features_file) is list:
            audio_file = []
            for j in range(len(features_file)):
                audio_file_j = features_file[j].replace(self.features_path, self.dataset.audio_path)
                audio_file_j = audio_file_j.replace('.npy', '.wav')    
                audio_file.append(audio_file_j)           
        #print(audio_file)
        return audio_file

    def convert_audio_path_to_features_path(self, audio_file):
        """ 
        Convert the path to an audio file (or list of files) to the path to the feature file.

        Parameters
        ----------
        audio_file : str or list of str
            Path to the audio file or files.

        Return
        ----------
        audio_file : str or list of str
            Path to the features file or files.

        """
        #print(audio_file)
        if type(audio_file) is str:
            features_file = audio_file.replace(self.dataset.audio_path, self.features_path)
            features_file = features_file.replace('.wav', '.npy')
        elif type(audio_file) is list:
            features_file = []
            for j in range(len(audio_file)): 
           #     print(audio_file[j])
            #    print(self.audio_path, self.features_path)
                features_file_j = audio_file[j].replace(self.dataset.audio_path, self.features_path)
                features_file_j = features_file_j.replace('.wav', '.npy')   
            #    print(features_file_j)
                features_file.append(features_file_j)                        
        #print(features_file)
        return features_file

    def extract_features(self):
        """ 
        Extracts features of each wav file present in self.audio_path.

        If the dataset is nos sampled at  the sampling rate set in
        feature_extractor.sr, the dataset is resampled before feature extraction.

        """

        # Change sampling rate if needed
        if not self.dataset.check_sampling_rate(self.feature_extractor.sr):
            self.dataset.change_sampling_rate(self.feature_extractor.sr)

        # Define path to audio and features folders
        audio_folder_sr = self.dataset.get_audio_path(self.feature_extractor.sr)

        # Check if the features were calculated already
        if not self.feature_extractor.check_features_folder(self.features_path):

            # Duplicate folder structure of audio in features folder
            duplicate_folder_structure(audio_folder_sr, self.features_path)

            # Navigate in the sctructure of audio folder and extract features 
            # of the each wav file
            for path_to_file_audio in list_wav_files_in_folder(audio_folder_sr):
                features_array = self.feature_extractor.calculate_features(
                    path_to_file_audio
                )
                path_to_features_file = path_to_file_audio.replace(
                    audio_folder_sr, self.features_path
                )
                path_to_features_file = path_to_features_file.replace(
                    'wav', 'npy'
                )
                np.save(path_to_features_file, features_array)

            # Save parameters.json for future checking
            self.feature_extractor.save_parameters_json(self.features_path)


    def check_if_features_extracted(self):
        """ 
        Check if the features were extracted before.

        """
        return self.feature_extractor.check_features_folder(self.features_path)
