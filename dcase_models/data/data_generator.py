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
    DataGenerator includes functions to load and manage DCASE datasets.
    This class can be redefined by child classes (see UrbanSound8k, ESC50)

    Attributes
    ----------
    dataset_path : str
        Path to the dataset folder
    features_folder : str
        Name of to the folder with the features files,
        default is 'features'
    feature_extractor : FeatureExtractor or childs
        Instance of FeatureExtractor. This is only needed to use
        functions related to feature extraction
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
    build()
        Define specific attributes of the dataset:
        label_list, fold_list, meta_file, etc.
    generate_file_lists()
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

    def __init__(self, dataset, feature_extractor, **kwargs):
    #, features_folder, features,
     #            audio_folder=None, use_validate_set=True):
        """ Initialize the DataGenerator
        
        Parameters
        ----------
        dataset_path : str
            Path to the dataset fold
        feature_extractor : FeatureExtractor or childs
            Instance of FeatureExtractor. This is only needed to use
            functions related to feature extraction

        """

        kwargs.setdefault('use_validate_set', True)
        kwargs.setdefault('features_folder', 'features')

        # General attributes
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.features_folder = kwargs['features_folder']
        self.use_validate_set = kwargs['use_validate_set']

        if feature_extractor.__class__.__bases__[0] is not FeatureExtractor:
            raise AttributeError('feature_extractor parent class is not FeatureExtractor')

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
            fold_val = get_fold_val(fold_test, self.fold_list)
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
                    if self.file_lists is not None:
                        Files_names_train.append(self.file_lists[fold_train][file])
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
                if self.file_lists is not None:
                    Files_names_test.append(self.file_lists[fold_test][file])
            X_test = np.concatenate(X_test, axis=0)
            Y_test = np.concatenate(Y_test, axis=0)

            return X_test, Y_test, Files_names_test
        



            # # take first sequence of each file
            # X_test_np = np.zeros((len(X_test), X_test[0].shape[1], X_test[0].shape[2]))
            # Y_test_np = np.zeros((len(X_test), Y_test[0].shape[1]))
            # for j in range(len(X_test)):
            #     ix = int(len(X)/2)
            #     X_test_np[j] = X_test[j][0]
            #     Y_test_np[j] = Y_test[j][0]



    def convert_features_path_to_audio_path(self, features_file):
        ''' convert ../features/foldX/MelSpectrogram/x.npy
            to ../audio/foldX/x.wav '''
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
        ''' convert ../features/foldX/MelSpectrogram/x.npy
            to ../audio/foldX/x.wav '''
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
        """ Extracts features of each wav file present in self.audio_folder.
        If the dataset is nos sampled at  the sampling rate set in
        feature_extractor.sr, the dataset is resampled before feature extraction.

        Parameters
        ----------
        feature_extractor : FeatureExtractor
            Instance of FeatureExtractor or childs

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
        # Check argument type

        if self.feature_extractor is None:
            raise AttributeError('''Can not load data without a FeatureExtractor. Init
                                 this object with a FeatureExtractor.''')    

        return self.feature_extractor.check_features_folder(self.features_path)





# class DataGenerator():
#     """
#     DataGenerator includes functions to load and manage DCASE datasets.
#     This class can be redefined by child classes (see UrbanSound8k, ESC50)

#     Attributes
#     ----------
#     dataset_path : str
#         Path to the dataset folder
#     features_folder : str
#         Name of to the folder with the features files,
#         default is 'features'
#     feature_extractor : FeatureExtractor or childs
#         Instance of FeatureExtractor. This is only needed to use
#         functions related to feature extraction
#     features : list of str
#         Names of features to be loaded
#     fold_list : list of str
#         List of fold names
#     label_list : list of str
#         List of label names
#     meta_file : str, optional
#         Path to the metadata file of dataset, default None
#     file_lists : dict
#         Dict of form {'fold1' : [file1, file2 ...], ...}
#     data : dict
#         Dict that contains the data.
#         Form: {'fold1' : {'X' : [features1, features2, ...],
#                           'Y': [ann1, ann2, ...]}, ...}

#     Methods
#     -------
#     build()
#         Define specific attributes of the dataset:
#         label_list, fold_list, meta_file, etc.
#     generate_file_lists()
#         Create self.file_lists, a dict thath includes a list of files per fold
#     get_annotations(file_name, features):
#         Returns the annotations of file in file_name path
#     data_generation(list_files_temp):
#         Returns features and annotations for all files in list_files_temp
#     load_data():
#         Creates self.data that contains features and annotations for all files
#         in all folds
#     get_data_for_training(fold_test, upsampling=False):
#         Returns arrays and lists for use in training process
#     get_data_for_testing(fold_test):
#         Returns lists to use to evaluate a model
#     get_one_example_per_file(fold_test)
#         Similar to get_data_for_training, but returns only one
#         example (sequence) for each file
#     return_file_list()
#         Returns self.file_lists
#     """

#     def __init__(self, dataset_path, feature_extractor=None, **kwargs):
#     #, features_folder, features,
#      #            audio_folder=None, use_validate_set=True):
#         """ Initialize the DataGenerator
        
#         Parameters
#         ----------
#         dataset_path : str
#             Path to the dataset fold
#         feature_extractor : FeatureExtractor or childs
#             Instance of FeatureExtractor. This is only needed to use
#             functions related to feature extraction

#         """

#         kwargs.setdefault('use_validate_set', True)
#         kwargs.setdefault('features_folder', 'features')

#         # General attributes
#         self.dataset_path = dataset_path
#         self.features_folder = kwargs['features_folder']
#         self.features_path = None
#         self.feature_extractor = feature_extractor
#         if feature_extractor is not None:
#             if feature_extractor.__class__.__bases__[0] is not FeatureExtractor:
#                 raise AttributeError('feature_extractor parent class is not FeatureExtractor')

#             feature_name = type(feature_extractor).__name__
#             mkdir_if_not_exists(os.path.join(dataset_path, self.features_folder))
#             self.features_path = os.path.join(dataset_path, self.features_folder, feature_name)  
#             mkdir_if_not_exists(self.features_path)
       
#         self.use_validate_set = kwargs['use_validate_set']

#         # Specific attributes
#         self.build()

#         # check if the dataset was download
#         # TODO improve this
#         # if not self.check_if_dataset_was_downloaded():
#         #   response = input('''The dataset was not downloaded : download [y]
#         #                      or continue without downloading [n] : ''')
#         #   if response == 'y':
#         #       self.download_dataset()

#         # make dataset folder if does not exists
#         # mkdir_if_not_exists(self.dataset_path)

#         # make features folder if does not exists
#         # if self.check_if_dataset_was_downloaded():
#         #    mkdir_if_not_exists(self.features_folder)

#         self.file_lists = {}
#         self.data = {}
        

#     def build(self):
#         """ Define specific parameters of the dataset
#         """        
#         self.fold_list = ["fold1", "fold2", "fold3", "fold4",
#                           "fold5", "fold6", "fold7", "fold8",
#                           "fold9", "fold10"]
#         self.label_list = ["air_conditioner", "car_horn", "children_playing",
#                            "dog_bark", "drilling", "engine_idling", "gun_shot",
#                            "jackhammer", "siren", "street_music"]
#         # self.meta_file = None
#         # self.taxonomy_file = None
#         self.evaluation_mode = 'cross-validation'
#         self.audio_folder = os.path.join(self.dataset_path, 'audio')

#         # self.folders_list = []
#         # for fold in self.fold_list:
#         #     audio_path = os.path.join(self.audio_folder, fold)
#         #     features_path = os.path.join(self.features_folder, fold)
#         #     audio_features_path = {'audio': audio_path,
#         #                            'features': features_path}
#         #     self.folders_list.append(audio_features_path)

#     def generate_file_lists(self):
#         """ Create self.file_lists, a dict thath includes a list of files per fold
#         """
#         for fold in self.fold_list:
#             features_folder = os.path.join(
#                 self.features_path, fold)
#             self.file_lists[fold] = sorted(
#                 glob.glob(os.path.join(features_folder, '*.npy')))

#     def get_annotations(self, file_name, features):
#         """ Returns the annotations of file in file_name path
#         Parameters
#         ----------
#         file_name : str
#             Path to the file
#         features : ndarray
#             3D array with the features of file_name

#         Returns
#         -------
#         ndarray
#             annotations of the file file_name

#         """
#         y = np.zeros((len(features), len(self.label_list)))
#         class_ix = int(os.path.basename(file_name).split('-')[1])
#         y[:, class_ix] = 1
#         return y

#     def data_generation(self, list_files_temp):
#         """ Returns features and annotations for all files in list_files_temp

#         Parameters
#         ----------
#         list_IDs_temp : list
#             List of file names

#         Return
#         ----------
#         features_list : list of ndarray
#             List of features for each file
#         annotations : list of ndarray
#             List of annotations matrix for each file

#         """
#         features_list = []
#         annotations = []

#         for file_name in list_files_temp:
#             features = np.load(file_name)
#             features_list.append(features)
#             y = self.get_annotations(file_name, features)
#             annotations.append(y)

#         return features_list, annotations

#     def load_data(self):
#         """ Creates self.data that contains features and annotations for all files
#         in all folds
#         """
#         if self.feature_extractor is None:
#             raise AttributeError('''Can not load data without a FeatureExtractor. Init
#                                  this object with a FeatureExtractor.''')

#         # Check argument type
#         if self.feature_extractor.__class__.__bases__[0] is not FeatureExtractor:
#             raise AttributeError('feature_extractor class is not FeatureExtractor')

#         if not self.check_if_features_extracted:
#             self.extract_features()

#         self.generate_file_lists()

#         for fold in progressbar(self.fold_list, prefix='fold: '):
#             X, Y = self.data_generation(self.file_lists[fold])
#             self.data[fold] = {'X': X, 'Y': Y}

#     def get_data_for_training(self, fold_test, upsampling=False):
#         """ Returns arrays and lists for use in training process

#         Parameters
#         ----------
#         fold_test : str
#             Name of fold test
#         upsampling : bool, optional
#             If true, the training set is upsampled to get a balanced array

#         Return
#         ----------
#         X_train : ndarray
#             3D array with all instances of train set.
#             Shape: (N_instances, N_hops, N_freqs)
#         Y_train : list of ndarray
#             2D array with annotations (one-hot coding)
#             Shape: (N_instances, N_classes)
#         X_val : list of ndarray
#             List of features for each file of validation set
#         Y_val : list of ndarray
#             List of annotations matrix for each file in validation set

#         """
#         # cross-validation mode
#         fold_val = get_fold_val(fold_test, self.fold_list)
#         folds_train = self.fold_list.copy()  # list(range(1,N_folds+1))
#         folds_train.remove(fold_test)
#         if self.use_validate_set:
#             folds_train.remove(fold_val)
#             X_val = self.data[fold_val]['X']
#             Y_val = self.data[fold_val]['Y']

#         X_train_list = []
#         Y_train_list = []
#         for fold in folds_train:
#             X_train_list.extend(self.data[fold]['X'])
#             Y_train_list.extend(self.data[fold]['Y'])

#         X_train = np.concatenate(X_train_list, axis=0)
#         Y_train = np.concatenate(Y_train_list, axis=0)

#         X_train_up = X_train.copy()
#         Y_train_up = Y_train.copy()

#         # upsampling
#         if upsampling:
#             n_classes = Y_train.shape[1]
#             Ns = np.zeros(n_classes)
#             for j in range(n_classes):
#                 Ns[j] = np.sum(Y_train[:, j] == 1)
#             Ns = np.floor(np.amax(Ns)/Ns)-1
#             for j in range(n_classes):
#                 if Ns[j] > 1:
#                     X_j = X_train[Y_train[:, j] == 1]
#                     Y_j = Y_train[Y_train[:, j] == 1]
#                     X_train_up = np.concatenate(
#                         [X_train_up]+[X_j]*int(Ns[j]), axis=0)
#                     Y_train_up = np.concatenate(
#                         [Y_train_up]+[Y_j]*int(Ns[j]), axis=0)

#         if self.use_validate_set:
#             return X_train_up, Y_train_up, X_val, Y_val
#         else:
#             return X_train_up, Y_train_up, X_train_list, Y_train_list

#     def get_data_for_testing(self, fold_test):
#         """ Returns lists to use to evaluate a model

#         Parameters
#         ----------
#         fold_test : str
#             Name of fold test

#         Return
#         ----------
#         X_test : list of ndarray
#             List of features for each file of test set
#         Y_test : list of ndarray
#             List of annotations matrix for each file in test set

#         """
#         # cross-validation mode
#         X_test = self.data[fold_test]['X'].copy()
#         Y_test = self.data[fold_test]['Y'].copy()

#         return X_test, Y_test

#     def get_one_example_per_file(self, fold_test, train=True):
#         # cross-validation mode

#         if train:
#             fold_val = get_fold_val(fold_test, self.fold_list)
#             folds_train = self.fold_list.copy()
#             folds_train.remove(fold_test)
#             folds_train.remove(fold_val)

#             # X_val = self.data[fold_val]['X']
#             # Y_val = self.data[fold_val]['Y']

#             X_train = []
#             Y_train = []
#             Files_names_train = []
#             for fold_train in folds_train:
#                 for file in range(len(self.data[fold_train]['X'])):
#                     X = self.data[fold_train]['X'][file]
#                     if len(X) == 0:
#                         continue
#                     #ix = int(len(X)/2)
#                     ix = int(len(X)/2) if len(X) > 1 else 0
#                     X = np.expand_dims(
#                         self.data[fold_train]['X'][file][ix], axis=0)
#                     X_train.append(X)
#                     Y = np.expand_dims(
#                         self.data[fold_train]['Y'][file][ix], axis=0)
#                     Y_train.append(Y)
#                     if self.file_lists is not None:
#                         Files_names_train.append(self.file_lists[fold_train][file])
#             X_train = np.concatenate(X_train, axis=0)
#             Y_train = np.concatenate(Y_train, axis=0)

#             return X_train, Y_train, Files_names_train
        
#         else:
#             X_test = []
#             Y_test = []
#             Files_names_test = []

#             for file in range(len(self.data[fold_test]['X'])):
#                 X = self.data[fold_test]['X'][file]
#                 if len(X) == 0:
#                     continue
#                 ix = int(len(X)/2) if len(X) > 1 else 0
#                 X = np.expand_dims(
#                     self.data[fold_test]['X'][file][ix], axis=0)
#                 X_test.append(X)
#                 Y = np.expand_dims(
#                     self.data[fold_test]['Y'][file][ix], axis=0)
#                 Y_test.append(Y)
#                 if self.file_lists is not None:
#                     Files_names_test.append(self.file_lists[fold_test][file])
#             X_test = np.concatenate(X_test, axis=0)
#             Y_test = np.concatenate(Y_test, axis=0)

#             return X_test, Y_test, Files_names_test
        



#             # # take first sequence of each file
#             # X_test_np = np.zeros((len(X_test), X_test[0].shape[1], X_test[0].shape[2]))
#             # Y_test_np = np.zeros((len(X_test), Y_test[0].shape[1]))
#             # for j in range(len(X_test)):
#             #     ix = int(len(X)/2)
#             #     X_test_np[j] = X_test[j][0]
#             #     Y_test_np[j] = Y_test[j][0]


#     def return_file_list(self, fold_test):
#         return self.file_lists[fold_test]

#     # def get_folder_lists(self):
#     #     return self.folders_list

#     def download_dataset(self, zenodo_url, zenodo_files):
#         if self.check_if_dataset_was_downloaded():
#             response = input(
#                 'The dataset was downloaded already: download again [y]' +
#                 ' or continue [n] : ')
#             if response == 'n':
#                 return None
#         download_files_and_unzip(self.dataset_path, zenodo_url, zenodo_files)

#     def set_dataset_download_finish(self):
#         log_file = os.path.join(self.dataset_path, 'download.txt')
#         with open(log_file, 'w') as txt_file:
#             txt_file.write('The dataset was download ...\n')

#     def check_if_dataset_was_downloaded(self):
#         log_file = os.path.join(self.dataset_path, 'download.txt')
#         return os.path.exists(log_file)

#     def convert_features_path_to_audio_path(self, features_file):
#         ''' convert ../features/foldX/MelSpectrogram/x.npy
#             to ../audio/foldX/x.wav '''
#         # audio_folder = self.audio_folder.split('/')[-1]
#         # features_folder = self.features_folder.split('/')[-1]
#         # audio_path = fetures_path.replace(self.features+'/', '')
#         # audio_path = audio_path.replace(features_folder, audio_folder)
#         # audio_path = audio_path.replace('.npy', '.wav')
#         print(features_file)
#         audio_file = features_file.replace(self.features_path, self.audio_folder)
#         audio_file = audio_file.replace('.npy', '.wav')
#         print(audio_file)
#         return audio_file

#     def change_sampling_rate(self, new_sr):
#         """ Changes sampling rate of each wav file in audio_folder.
#         Creates a new folder named audio_folder{new_sr} (i.e audio22050)
#         and converts each wav file in audio_folder and save the result in 
#         the new folder. 

#         Parameters
#         ----------
#         sr : int
#             Sampling rate

#         """

#         new_audio_folder = self.audio_folder + str(new_sr)
#         duplicate_folder_structure(self.audio_folder, new_audio_folder)

#         tfm = sox.Transformer()
#         tfm.convert(samplerate=new_sr)

#         for path_to_file in list_wav_files_in_folder(self.audio_folder):
#             path_to_destination = path_to_file.replace(self.audio_folder, new_audio_folder)
#             #print(path_to_destination)
#             if os.path.exists(path_to_destination):
#                 continue
#             tfm.build(path_to_file, path_to_destination)

#     def check_sampling_rate(self, sr):
#         """ Checks if dataset was resampled before. 
#         For now, only checks if the folder dataset_path/audio{sr} exists and
#         each wav file present in dataset_path/audio is present in 
#         dataset_path/audio{sr}. Does not checks if the audio files were resampled
#         correctly.

#         Parameters
#         ----------
#         sr : int
#             Sampling rate

#         Returns
#         ----------
#         bool
#             True if the dataset was resampled before

#         """

#         audio_folder_sr = self.audio_folder + str(sr)
#         if not os.path.exists(audio_folder_sr):
#             return False

#         for path_to_file in list_wav_files_in_folder(self.audio_folder):
#             path_to_destination = path_to_file.replace(self.audio_folder, audio_folder_sr)
#             # TODO: check if the audio file was resampled correctly, not only if exits.
#             if not os.path.exists(path_to_destination):
#                 return False
            
#         return True

#     def extract_features(self):
#         """ Extracts features of each wav file present in self.audio_folder.
#         If the dataset is nos sampled at  the sampling rate set in
#         feature_extractor.sr, the dataset is resampled before feature extraction.

#         Parameters
#         ----------
#         feature_extractor : FeatureExtractor
#             Instance of FeatureExtractor or childs

#         """

#         if self.feature_extractor is None:
#             raise AttributeError('''Can not load data without a FeatureExtractor. Init
#                                  this object with a FeatureExtractor.''')

#         # Change sampling rate if needed
#         if not self.check_sampling_rate(self.feature_extractor.sr):
#             self.change_sampling_rate(self.feature_extractor.sr)

#         # Define path to audio and features folders
#         audio_folder_sr = self.audio_folder + str(self.feature_extractor.sr)

#         # Check if the features were calculated already
#         if not self.feature_extractor.check_features_folder(self.features_path):

#             # Duplicate folder structure of audio in features folder
#             duplicate_folder_structure(audio_folder_sr, self.features_path)

#             # Navigate in the sctructure of audio folder and extract features 
#             # of the each wav file
#             for path_to_file_audio in list_wav_files_in_folder(audio_folder_sr):
#                 features_array = self.feature_extractor.calculate_features(
#                     path_to_file_audio
#                 )
#                 path_to_features_file = path_to_file_audio.replace(
#                     audio_folder_sr, self.features_path
#                 )
#                 path_to_features_file = path_to_features_file.replace(
#                     'wav', 'npy'
#                 )
#                 np.save(path_to_features_file, features_array)

#             # Save parameters.json for future checking
#             self.feature_extractor.save_parameters_json(self.features_path)


#     def check_if_features_extracted(self):
#         # Check argument type

#         if self.feature_extractor is None:
#             raise AttributeError('''Can not load data without a FeatureExtractor. Init
#                                  this object with a FeatureExtractor.''')    

#         return self.feature_extractor.check_features_folder(self.features_path)
