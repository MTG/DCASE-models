import os
import sox
import soundfile as sf
import numpy as np
from librosa.core import db_to_power, power_to_db

from dcase_models.data.dataset_base import Dataset
from dcase_models.util.files import duplicate_folder_structure
from dcase_models.util.files import list_wav_files
from dcase_models.util.ui import progressbar


class WhiteNoise():
    """ Implements white noise augmentation.

    The structure is similar to sox.Transformer in order to keep
    compatibility with sox.

    Parameters
    ----------
    snr : float
        Signal to noise ratio.

    """
    def __init__(self, snr):
        """ Initialize the white noise.
        """
        self.snr = snr

    def build(self, file_origin, file_destination):
        """ Add noise to the file_origin and save the result in file_destination.

        Parameters
        ----------
        file_origin : str
            Path to the source file.
        file_destination : str
            Path to the destination file.

        """
        audio, sr = sf.read(file_origin)
        # Calculate signal mean power
        s_power = np.mean(audio**2)
        s_db = power_to_db(s_power)

        # Get noise power
        n_db = s_db - self.snr
        n_power = db_to_power(n_db)

        # Define noise signal
        noise = np.random.normal(
            loc=0.0, scale=np.sqrt(n_power), size=audio.shape
        )

        # Sum noise
        aug_audio = audio + noise

        # Check if the new singal clipped
        if np.any(aug_audio > 1.0):
            # TODO: check this solution
            aug_audio = aug_audio/np.amax(aug_audio)

        # Save result to file
        sf.write(file_destination, aug_audio, sr)


class AugmentedDataset(Dataset):
    """ Class that manage data augmentation.

    Basically, it takes an instance of Dataset and generates an augmented one.
    Includes methods to generate data augmented versions of the audio files
    in an existing Dataset.

    Parameters
    ----------
    dataset : Dataset
        Instance of Dataset to be augmented.
    augmentations_list : list
        List of augmentation types and their parameters.
        Dict of form: [{'type' : aug_type, 'param1': param1 ...} ...].
        e.g.::

            [
                {'type': 'pitch_shift', 'n_semitones': -1},
                {'type': 'time_stretching', 'factor': 1.05}
            ]

    sr : int
        Sampling rate

    Examples
    --------
    Define an instance of UrbanSound8k and convert it into an augmented
    instance of the dataset. Note that the actual augmentation is performed
    when process() method is called.

    >>> from dcase_models.data.datasets import UrbanSound8k
    >>> from dcase_models.data.data_augmentation import AugmentedDataset
    >>> dataset = UrbanSound8k('../datasets/UrbanSound8K')
    >>> augmentations = [
            {"type": "pitch_shift", "n_semitones": -1},
            {"type": "time_stretching", "factor": 1.05},
            {"type": "white_noise", "snr": 60}
        ]
    >>> aug_dataset = AugmentedDataset(dataset, augmentations)
    >>> aug_dataset.process()

    """

    def __init__(self, dataset, sr,
                 augmentations_list):
        """ Initialize the AugmentedDataset.

        Initialize sox Transformers for each type of augmentation.
        """

        self.dataset = dataset
        self.augmentations_list = augmentations_list
        self.sr = sr

        # Init sox Transformers
        # Append these to the self.augmentations_list as a new
        # augmentation property.

        for index in range(len(augmentations_list)):
            augmentation = augmentations_list[index]
            aug_type = augmentation['type']
            tfm = sox.Transformer()
            if aug_type == 'pitch_shift':
                tfm.pitch(augmentation['n_semitones'])
            if aug_type == 'time_stretching':
                tfm.tempo(augmentation['factor'])
                # tfm.stretch(augmentation['factor'])
            if aug_type == 'white_noise':
                tfm = WhiteNoise(augmentation['snr'])
            augmentations_list[index]['transformer'] = tfm

        # Copy attributes of dataset
        self.__dict__.update(dataset.__dict__)

    def get_annotations(self, file_path, features, time_resolution):
        return self.dataset.get_annotations(file_path, features, time_resolution)

    def generate_file_lists(self):
        """ Create self.file_lists, a dict that includes a list of files per fold.

        Just call dataset.generate_file_lists() and copy the attribute.

        """
        self.dataset.generate_file_lists()
        self.file_lists = self.dataset.file_lists.copy()

    def process(self):
        """ Generate augmentated data for each file in dataset.

        Replicate the folder structure of {DATASET_PATH}/audio/original
        into the folder of each augmentation folder.

        """
        if not self.dataset.check_sampling_rate(self.sr):
            print("Changing sampling rate ...")
            self.dataset.change_sampling_rate(self.sr)
            print('Done!')

        # Get path to the original audio files and list of
        # folders with augmented files.
        _, sub_folders = self.get_audio_paths(self.sr)
        path_original = sub_folders[0]
        paths_augments = sub_folders[1:]

        for index in range(len(self.augmentations_list)):
            augmentation = self.augmentations_list[index]
            path_augmented = paths_augments[index]

            # Replicate folder structure of the original files into
            # the augmented folder.
            duplicate_folder_structure(path_original, path_augmented)
            # Process each file in path_original
            for path_to_file in progressbar(list_wav_files(path_original)):
                path_to_destination = path_to_file.replace(
                    path_original, path_augmented
                )
                if os.path.exists(path_to_destination):
                    continue
                augmentation['transformer'].build(
                    path_to_file, path_to_destination
                )

    def get_audio_paths(self, sr=None):
        """ Returns a list of paths to the folders that include the dataset
        augmented files.

        The folder of each augmentation is defined using its name and
        parameter values.

        e.g. {DATASET_PATH}/audio/pitch_shift_1 where 1 is the 'n_semitones'
        parameter.

        Parameters
        ----------
        sr : int or None, optional
            Sampling rate (optional). We keep this parameter to keep
            compatibility with Dataset.get_audio_paths() method.

        Returns
        -------
        audio_path : str
            Path to the root audio folder.
            e.g. DATASET_PATH/audio
        subfolders : list of str
            List of subfolders include in audio folder.
            e.g.::

                [
                    '{DATASET_PATH}/audio/original',
                    '{DATASET_PATH}/audio/pitch_shift_1',
                    '{DATASET_PATH}/audio/time_stretching_1.1',
                ]

        """
        if sr is not None:
            audio_path = self.audio_path + str(sr)
        else:
            audio_path = self.audio_path
        subfolders = [os.path.join(audio_path, 'original')]

        for augmentation in self.augmentations_list:
            aug_type = augmentation['type']
            if aug_type == 'pitch_shift':
                aug_folder = 'pitch_shift_%d' % augmentation['n_semitones']
            if aug_type == 'time_stretching':
                aug_folder = 'time_stretching_%2.2f' % augmentation['factor']
            if aug_type == 'white_noise':
                aug_folder = 'white_noise_%2.2f' % augmentation['snr']
            subfolders.append(os.path.join(audio_path, aug_folder))

        return audio_path, subfolders
