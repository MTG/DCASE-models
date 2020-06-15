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

class Dataset():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.file_lists = {}
        self.build()

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.fold_list = ['fold1', 'fold2', 'fold3']
        self.label_list = ['class1', 'class2', 'class3']

    def generate_file_lists(self):
        """ Create self.file_lists, a dict thath includes a list of files per fold
        """
        for fold in self.fold_list:
            audio_folder = os.path.join(
                self.audio_path, fold)
            self.file_lists[fold] = sorted(
                glob.glob(os.path.join(audio_folder, '*.wav')))

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


    def download_dataset(self, zenodo_url, zenodo_files):
        if self.check_if_dataset_was_downloaded():
            response = input(
                'The dataset was downloaded already: download again [y]' +
                ' or continue [n] : ')
            if response == 'n':
                return None
        download_files_and_unzip(self.dataset_path, zenodo_url, zenodo_files)
        return True

    def set_dataset_download_finish(self):
        log_file = os.path.join(self.dataset_path, 'download.txt')
        with open(log_file, 'w') as txt_file:
            txt_file.write('The dataset was download ...\n')

    def check_if_dataset_was_downloaded(self):
        log_file = os.path.join(self.dataset_path, 'download.txt')
        return os.path.exists(log_file)

    def get_audio_path(self, sr=None):
        if sr is None:
            audio_path = self.audio_path
        else:
            audio_path = self.audio_path + str(sr) 
        return audio_path

    def change_sampling_rate(self, new_sr):
        """ Changes sampling rate of each wav file in audio_folder.
        Creates a new folder named audio_folder{new_sr} (i.e audio22050)
        and converts each wav file in audio_folder and save the result in 
        the new folder. 

        Parameters
        ----------
        sr : int
            Sampling rate

        """

        new_audio_folder = self.get_audio_path(new_sr)
        duplicate_folder_structure(self.audio_path, new_audio_folder)

        tfm = sox.Transformer()
        tfm.convert(samplerate=new_sr)

        for path_to_file in list_wav_files_in_folder(self.audio_path):
            path_to_destination = path_to_file.replace(self.audio_path, new_audio_folder)
            #print(path_to_destination)
            if os.path.exists(path_to_destination):
                continue
            tfm.build(path_to_file, path_to_destination)

    def check_sampling_rate(self, sr):
        """ Checks if dataset was resampled before. 
        For now, only checks if the folder dataset_path/audio{sr} exists and
        each wav file present in dataset_path/audio is present in 
        dataset_path/audio{sr}. Does not checks if the audio files were resampled
        correctly.

        Parameters
        ----------
        sr : int
            Sampling rate

        Returns
        ----------
        bool
            True if the dataset was resampled before

        """

        audio_folder_sr = self.get_audio_path(sr)
        if not os.path.exists(audio_folder_sr):
            return False

        for path_to_file in list_wav_files_in_folder(self.audio_path):
            path_to_destination = path_to_file.replace(self.audio_path, audio_folder_sr)
            # TODO: check if the audio file was resampled correctly, not only if exits.
            if not os.path.exists(path_to_destination):
                return False
            
        return True
