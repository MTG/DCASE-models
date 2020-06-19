import os
import sox

from ..utils.files import download_files_and_unzip
from ..utils.files import duplicate_folder_structure
from ..utils.files import list_wav_files


class Dataset():
    """
    Abstract base class to load and manage DCASE datasets.

    This class can be redefined by inheritance (see UrbanSound8k, ESC50)

    Attributes
    ----------
    dataset_path : str
        Path to the dataset folder.
    file_lists : dict
        This dictionary stores the list of files for each fold.
        Dict of form: {fold_name : list_of_files}.
        e.g. {'fold1' : [file1, file2 ...], 'fold2' : [file3, file4 ...],}.
    audio_path : str
        Path to the audio folder, i.e {self.dataset_path}/audio.
        Define in build()
    fold_list : list
        List of fold names.
        e.g. ['fold1', 'fold2', 'fold3', ...]
    label_list : list
        List of class labels:
        e.g. ['dog', 'siren', ...]

    Methods
    -------
    build()
        Define specific attributes of the dataset:
        label_list, fold_list, meta_file, etc.
    generate_file_lists()
        Create self.file_lists, a dict that stores a list of files per fold.
    get_annotations(file_name, features):
        Return the annotations of the file in file_path.
    download(self, zenodo_url, zenodo_files):
        Download and decompress the dataset from zenodo.
    set_as_downloaded():
        Save a download.txt file in dataset_path as a downloaded flag
    check_if_downloaded():
        Check if the dataset was downloaded.
        Just check if exists download.txt file.
    get_audio_paths(sr=None)
        Return path to the audio folder.
        If sr is not None, return {audio_path}{sr}
    change_sampling_rate(new_sr)
        Changes sampling rate of each wav file in audio_path.
    check_sampling_rate(new_sr)
        Check if the dataset was resampled to new_sr.
    """
    def __init__(self, dataset_path):
        """
        Init Dataset
        """

        self.dataset_path = dataset_path
        self.file_lists = {}
        self.build()

    def build(self):
        """
        Build the dataset.

        Define specific attributes of the dataset.
        It's mandatory to define audio_path, fold_list and label_list.
        Other attributes may be defined here (url, authors, etc.).

        """

        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.fold_list = ['fold1', 'fold2', 'fold3']
        self.label_list = ['class1', 'class2', 'class3']

    def generate_file_lists(self):
        """
        Create self.file_lists, a dict that includes a list of files per fold.

        Each dataset has a different way of organizing the files. This
        function defines the dataset structure.
        """

        # Creates an empty dict
        self.file_lists = {[] for fold in self.fold_list}

    def get_annotations(self, file_path, features):
        """
        Return the annotations of the file in file_path.

        Parameters
        ----------
        file_path : str
            Path to the file
        features : ndarray
            nD array with the features of file_path

        Returns
        -------
        ndarray
            Annotations of the file file_path
            Expected output shape: (features.shape[0], len(self.label_list))

        """
        pass

    def download(self, zenodo_url, zenodo_files, force_download=False):
        """
        Download and decompress the dataset from zenodo.

        Parameters
        ----------
        zenodo_url : str
            URL with the zenodo files.
            e.g. 'https://zenodo.org/record/12345/files'
        zenodo_files : list of str
            List of files.
            e.g. ['file1.tar.gz', 'file2.tar.gz', 'file3.tar.gz']
        force_download : bool
            If True, download the dataset even if was downloaded before.

        Returns
        -------
        bool
            True if the downloading process was successful.

        """
        if self.check_if_downloaded() and not force_download:
            return False

        download_files_and_unzip(self.dataset_path, zenodo_url, zenodo_files)
        return True

    def set_as_downloaded(self):
        """
        Save a download.txt file in dataset_path as a downloaded flag.
        """
        log_file = os.path.join(self.dataset_path, 'download.txt')
        with open(log_file, 'w') as txt_file:
            # pass
            txt_file.write('')

    def check_if_downloaded(self):
        """
        Check if the dataset was downloaded.

        Just check if exists download.txt file.

        Further checks in the future.

        """
        log_file = os.path.join(self.dataset_path, 'download.txt')
        return os.path.exists(log_file)

    def get_audio_paths(self, sr=None):
        """
        Return paths to the audio folder.

        If sr is None, return audio_path. Else, return {audio_path}{sr}.

        Parameters
        ----------
        sr : int or None, optional
            Sampling rate.

        Returns
        -------
        audio_path : str
            Path to the root audio folder.
            e.g. DATASET_PATH/audio
        subfolders : list of str
            List of subfolders include in audio folder.
            Important when use AugmentedDataset.
            e.g. ['{DATASET_PATH}/audio/original']
        """
        subfolders = None
        if sr is None:
            audio_path = self.audio_path
        else:
            audio_path = self.audio_path + str(sr)
            subfolders = [os.path.join(audio_path, 'original')]

        return audio_path, subfolders

    def change_sampling_rate(self, new_sr):
        """
        Change sampling rate of each wav file in audio_path.

        Creates a new folder named audio_path{new_sr} (i.e audio22050)
        and converts each wav file in audio_path and save the result in
        the new folder.

        Parameters
        ----------
        sr : int
            Sampling rate.

        """
        new_audio_path, subfolders = self.get_audio_paths(new_sr)
        new_audio_folder = subfolders[0]  # audio22050/original
        duplicate_folder_structure(self.audio_path, new_audio_folder)

        tfm = sox.Transformer()
        tfm.convert(samplerate=new_sr)

        for path_to_file in list_wav_files(self.audio_path):
            path_to_destination = path_to_file.replace(
                self.audio_path, new_audio_folder
            )
            if os.path.exists(path_to_destination):
                continue
            tfm.build(path_to_file, path_to_destination)

    def check_sampling_rate(self, sr):
        """
        Check if dataset was resampled before.

        For now, only check if the folder {audio_path}{sr} exists and
        each wav file present in audio_path is also present in
        {audio_path}{sr}.

        Parameters
        ----------
        sr : int
            Sampling rate.

        Returns
        ----------
        bool
            True if the dataset was resampled before.

        """

        audio_folder_sr = self.get_audio_paths(sr)[0]
        if not os.path.exists(audio_folder_sr):
            return False

        for path_to_file in list_wav_files(self.audio_path):
            path_to_destination = path_to_file.replace(
                self.audio_path, audio_folder_sr
            )
            # TODO: check if the audio file was resampled correctly,
            # not only if exists.
            if not os.path.exists(path_to_destination):
                return False
        return True
