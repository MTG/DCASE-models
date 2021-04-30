import os
import numpy as np
import inspect
import random

import tensorflow as tf
tensorflow2 = tf.__version__.split('.')[0] == '2'

if tensorflow2:
    from tensorflow.keras.utils import Sequence
else:
    from keras.utils import Sequence

from dcase_models.data.feature_extractor import FeatureExtractor
from dcase_models.data.dataset_base import Dataset
# from .data_augmentation import AugmentedDataset


class DataGenerator():
    """ Includes methods to load features files from DCASE datasets.

    Parameters
    ----------
    dataset : Dataset
        Instance of the Dataset used to load the data. Note that the dataset
        has to be downloaded before initializing the DataGenerator.
        Refer to dcase-models/data/datasets.py for a complete list of available
        datasets.

    inputs : instance of FeatureExtractor or list of FeatureExtractor instances
        Instance(s) of FeatureExtractor. These are the feature extractor(s) used
        to generate the features.
        For multi-input, pass a list of FeatureExtractor instances.

    folds : list of str
        List of folds to be loaded. Each fold has to be in dataset.fold_list.
        Note that since the folds used at each stage of the pipeline
        (training, validation, evaluation) are different, an instance of
        DataGenerator for each stage has to be created.
        e.g. ['fold1', 'fold2', 'fold3', ...]

    outputs : str, FeatureExtractor or list, default='annotations'
        Instance(s) of FeatureExtractor used to generate the outputs.
        To use the annotations obtained from Dataset, use a string.
        For multi-output, use a list of FeatureExtractor and/or strings.

    batch_size : int, default=32
        Number of files loaded when call get_data_batch().
        Note that the meaning of batch_size here is slightly different from 
        the one in machine learning libraries like keras. In these libraries
        batch_size means the number of instances (sequences in DCASE-models)
        used in each training step. Here batch_size is the number of files,
        and therefore, the number of sequences varies in each batch.

    shuffle: bool, default=True
        When training a model, it is typical to shuffle the dataset at the end
        of each epoch. If shuffle is True (default), then the audio file list
        is shuffled when the class is initialized and when shuffle_list()
        method is called.

    train : bool, default True
        When training, it is typical to feed the model with a numpy array
        that contains all the data concatenated. For validation and
        testing it is necessary to have the features of each file
        separate in order to do a file-wise evaluation.
        Therefore, if train is True, the loaded data is concatenated and
        converted to a numpy array. If train is False get_data() and
        get_data_batch() return a list, whose elements are the features
        of each file in the audio_file_list.

    scaler : Scaler or None, default=None
        If is not None, the Scaler object is used to scale the data
        after loading.

    scaler_outputs : Scaler or None, default=None
        Same as scaler but for the system outputs.

    Attributes
    ----------
    audio_file_list : list of dict
        List of audio files from which the features will be loaded.
        Each element in the list includes information of the original
        audio file (important to get the annotations) and the subfolder where
        is the resampled (and maybe augmented) audio file.
        e.g.:

            audio_file_list = [
                {'file_original': 'audio/1.wav', 'sub_folder': 'original'},
                {'file_original': 'audio/1.wav', 'sub_folder': 'pitch_shift_1'},
                {'file_original': 'audio/2.wav', 'sub_folder': 'original'},
                ...
            ]


    See Also
    --------
    Dataset : Dataset class

    FeatureExtractor : FeatureExtractor class


    Examples
    --------
    Create instances of Dataset and FeatureExtractor with default parameters

    >>> from dcase_models.data.datasets import UrbanSound8k
    >>> from dcase_models.data.features import MelSpectrogram
    >>> from dcase_models.data.data_generator import DataGenerator
    >>> dataset = UrbanSound8k('../datasets/UrbanSound8k')
    >>> features = MelSpectrogram()

    Assuming that the dataset was downloaded and features were extracted
    already, we can initialize the data generators. This example uses fold1 
    and fold2 for training and fold3 for validation.

    >>> data_gen_train = DataGenerator(
        dataset, features, ['fold1', 'fold2'], train=True)
    >>> data_gen_val = DataGenerator(
        dataset, features, ['fold3'], train=False)

    >>> X_train, Y_train = data_gen_train.get_data_batch(0)
    >>> print(X_train.shape, Y_train.shape)
        (212, 43, 64) (212, 10)

    >>> X_val, Y_val = data_gen_val.get_data_batch(0)
    >>> print(len(X_val), len(Y_val))
        32 32
    >>> print(X_val[0].shape, Y_val[0].shape)
        (7, 43, 64) (7, 10)

    >>> X_train, Y_train = data_gen_train.get_data()
    >>> print(X_train.shape, Y_train.shape)
        (11095, 43, 64) (11095, 10)

    >>> X_val, Y_val = data_gen_val.get_data()
    >>> print(len(X_val), len(Y_val))
        925 925
    >>> print(X_val[0].shape, Y_val[0].shape)
        (7, 43, 64) (7, 10)

    """
    def __init__(self, dataset, inputs, folds,
                 outputs='annotations',
                 batch_size=32, shuffle=True,
                 train=True, scaler=None, scaler_outputs=None):
        """ Initialize the DataGenerator.

        Generates the audio_file_list by concatenating all the files
        from the folds passed as an argument.
        """
        # General attributes
        self.dataset = dataset
        self.inputs = inputs
        if type(inputs) != list:
            self.inputs = [inputs]
        self.folds = folds
        self.outputs = outputs
        if type(outputs) != list:
            self.outputs = [outputs]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.scaler = scaler
        self.scaler_outputs = scaler_outputs

        if (Dataset not in inspect.getmro(dataset.__class__)):
            raise AttributeError(
                'dataset has to be an instance of Dataset or similar'
            )

        if (not dataset.check_if_downloaded() and
           dataset.__class__ is not Dataset):
            raise AttributeError(
                ('The dataset was not downloaded. Please download it '
                 'before using DataGenerator')
            )

        for j, inp in enumerate(self.inputs):
            if ((FeatureExtractor not in inspect.getmro(inp.__class__)) and
               (type(inp) is not str)):
                raise AttributeError(('Each input has to be an '
                                      'instance of FeatureExtractor '
                                      'or similar'))
            # TODO: Check if all inputs share sr
            # TODO: Check if str is available in dataset

            if (not inp.check_if_extracted(dataset) and
               inp.__class__ is not FeatureExtractor):
                raise AttributeError(
                    ('Features were not extracted '
                     'for input: %d - %s' % (j, inp.__class__.__name__))
                )

            if FeatureExtractor in inspect.getmro(inp.__class__):
                self.sr = inp.sr
                self.time_resolution = inp.sequence_hop_time

        for output in self.outputs:
            if ((FeatureExtractor not in inspect.getmro(output.__class__)) and
               (type(output) is not str)):
                raise AttributeError('''Each input has to be an
                                        instance of FeatureExtractor
                                        or similar''')

        # self.features_file_list = []
        self.audio_file_list = []

        # Get audio paths
        self.dataset.generate_file_lists()
        audio_path, subfolders = self.dataset.get_audio_paths(
            self.sr
        )

        if not train:
            # If not train, don't use augmentation
            subfolders = [subfolders[0]]
        for fold in folds:
            for subfolder in subfolders:
                subfolder_name = os.path.basename(subfolder)
                files_audio = self.dataset.file_lists[fold]
                for file_audio in files_audio:
                    self.audio_file_list.append(
                        {'file_original': file_audio,
                         'sub_folder': subfolder_name})
                # file_features = self.convert_audio_path_to_features_path(
                #     files_audio, subfolder=subfolder_name
                # )
                # self.features_file_list.extend(file_features)

        if shuffle:
            self.shuffle_list()

        self.data = {}

    def _data_generation(self, list_files):
        """ Returns features and annotations for all files in list_files.

        Parameters
        ----------
        list_files : list of str
            List of file paths.

        Returns
        -------
        features_list : list of ndarray
            List of features for each file.
        annotations : list of ndarray
            List of annotations matrix for each file.

        """
        inputs_lists = [[] for _ in range(len(self.inputs))]
        outputs_lists = [[] for _ in range(len(self.outputs))]

        for file_dict in list_files:
            file_original = file_dict['file_original']
            sub_folder = file_dict['sub_folder']

            for j, input in enumerate(self.inputs):
                if type(input) is not str:
                    features_path = input.get_features_path(self.dataset)
                    file_features = self.convert_audio_path_to_features_path(
                        file_original, features_path, subfolder=sub_folder)
                    features = np.load(file_features)
                    inputs_lists[j].append(features)
                else:
                    raise AttributeError('Not available')
                    # TODO: ADD this option

            for j, output in enumerate(self.outputs):
                if type(output) is not str:
                  #  print(output)
                    features_path = output.get_features_path(self.dataset)
                    file_features = self.convert_audio_path_to_features_path(
                        file_original, features_path, subfolder=sub_folder)
                    features = np.load(file_features)
                   # print(features.shape)
                    outputs_lists[j].append(features)
                else:
                    # TODO: Add option to other outputs
                    y = self.dataset.get_annotations(
                        file_original, inputs_lists[0][-1],
                        self.time_resolution)
                    outputs_lists[j].append(y)
                    # TODO: Improve how we pass features array to get_ann..

        return inputs_lists, outputs_lists

    def get_data(self):
        """ Return all data from the selected folds.

        If train were set as True, the output is concatenated and
        converted to a numpy array. Otherwise the outputs are lists whose
        elements are the features of each file.

        Returns
        -------
        X : list or ndarray
            List or array of features for each file.
        Y : list or ndarray
            List or array of annotations for each file.

        """
        X_list, Y_list = self._data_generation(self.audio_file_list)

        if self.scaler is not None:
            X_list = self.scaler.transform(X_list)
        if self.scaler_outputs is not None:
            Y_list = self.scaler_outputs.transform(Y_list)

        X = [[] for _ in range(len(self.inputs))]
        Y = [[] for _ in range(len(self.outputs))]

        for j in range(len(self.inputs)):
            if self.train:
                X[j] = np.concatenate(X_list[j], axis=0)
            else:
                X[j] = X_list[j].copy()

        for j in range(len(self.outputs)):
            if self.train:
                Y[j] = np.concatenate(Y_list[j], axis=0)
            else:
                Y[j] = Y_list[j].copy()

        if len(X) == 1:
            X = X[0]
        if len(Y) == 1:
            Y = Y[0]

        return X, Y

    def get_data_batch(self, index):
        """ Return the data from the batch given by argument.

        If train were set as True, the output is concatenated and
        converted to a numpy array. Otherwise the outputs are lists whose
        elements are the features of each file.

        Returns
        -------
        X : list or ndarray
            List or array of features for each file.
        Y : list or ndarray
            List or array of annotations for each file.

        """
        list_file_batch = self.audio_file_list[
            index*self.batch_size:(index+1)*self.batch_size
        ]
        # Generate data
        X_list, Y_list = self._data_generation(list_file_batch)

        if self.scaler is not None:
            X_list = self.scaler.transform(X_list)
        if self.scaler_outputs is not None:
            Y_list = self.scaler_outputs.transform(Y_list)

        X = [[] for _ in range(len(self.inputs))]
        Y = [[] for _ in range(len(self.outputs))]

        for j in range(len(self.inputs)):
            if self.train:
                X[j] = np.concatenate(X_list[j], axis=0)
            else:
                X[j] = X_list[j].copy()

        for j in range(len(self.outputs)):
            if self.train:
                Y[j] = np.concatenate(Y_list[j], axis=0)
            else:
                Y[j] = Y_list[j].copy()

        if len(X) == 1:
            X = X[0]
        if len(Y) == 1:
            Y = Y[0]

        return X, Y

    def get_data_from_file(self, file_index):
        """ Returns the data from the file index given by argument.

        Returns
        -------
        X : ndarray
            Array of features for each file.
        Y : ndarray
            Array of annotations for each file.

        """
        # Generate data
        X, Y = self._data_generation([self.audio_file_list[file_index]])
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.scaler_outputs is not None:
            Y = self.scaler_outputs.transform(Y)

        if len(X) == 1:
            X = X[0]
        if len(Y) == 1:
            Y = Y[0]

        return X[0].copy(), Y[0].copy()

    def convert_features_path_to_audio_path(self, features_file,
                                            features_path, sr=None):
        """ Converts features path(s) to audio path(s).

        Parameters
        ----------
        features_file : str or list of str
            Path(s) to the features file(s).

        Returns
        -------
        audio_file : str or list of str
            Path(s) to the audio file(s).

        """
        audio_path, _ = self.dataset.get_audio_paths(sr=sr)

        if type(features_file) is str:
            audio_file = features_file.replace(
                features_path, audio_path
            )
            audio_file = audio_file.replace('.npy', '.wav')
        elif type(features_file) is list:
            audio_file = []
            for j in range(len(features_file)):
                audio_file_j = features_file[j].replace(
                    features_path, audio_path
                )
                audio_file_j = audio_file_j.replace('.npy', '.wav')
                audio_file.append(audio_file_j)
        return audio_file

    def convert_audio_path_to_features_path(self, audio_file,
                                            features_path, subfolder=''):
        """ Converts audio path(s) to features path(s).

        Parameters
        ----------
        audio_file : str or list of str
            Path(s) to the audio file(s).

        Returns
        -------
        features_file : str or list of str
            Path(s) to the features file(s).

        """
        if subfolder != '':
            features_path = os.path.join(features_path, subfolder)
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
        """ Removes the subfolder string related to augmentation from a path.

        Converts DATASET_PATH/audio/original/... into DATASET_PATH/audio/...

        Parameters
        ----------
        path : str or list of str
            Path to be converted.

        Returns
        -------
        features_file : str or list of str
            Path(s) to the features file(s).

        """
        audio_path, subfolders = self.dataset.get_audio_paths()
        audio_path_sr, subfolders_sr = self.dataset.get_audio_paths()
        new_path = None
        for subfolder in subfolders:
            if subfolder in path:
                new_path = path.replace(subfolder, audio_path)
                break

        return new_path

    def shuffle_list(self):
        """ Shuffles features_file_list.

        Notes
        -----
        Only shuffle the list if shuffle is True.

        """
        if self.shuffle:
            random.shuffle(self.audio_file_list)

    def __len__(self):
        """ Get the number of batches.

        """
        return int(np.ceil(len(self.audio_file_list) / self.batch_size))

    def set_scaler(self, scaler):
        """ Set scaler object.

        """
        self.scaler = scaler

    def set_scaler_outputs(self, scaler_outputs):
        """ Set scaler object.

        """
        self.scaler_outputs = scaler_outputs


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
