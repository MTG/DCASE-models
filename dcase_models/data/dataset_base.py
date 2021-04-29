import os
import sox

from dcase_models.util.files import download_files_and_unzip
from dcase_models.util.files import duplicate_folder_structure
from dcase_models.util.files import list_wav_files, list_all_files
from dcase_models.util.ui import progressbar


class Dataset():
    """ Abstract base class to load and manage DCASE datasets.

    Descendants of this class are defined to manage specific
    DCASE databases (see UrbanSound8k, ESC50)

    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/UrbanSound8k).

    Attributes
    ----------
    file_lists : dict
        This dictionary stores the list of files for each fold.
        Dict of form: {fold_name : list_of_files}.
        e.g. {'fold1' : [file1, file2 ...], 'fold2' : [file3, file4 ...],}.
    audio_path : str
        Path to the audio folder, i.e {self.dataset_path}/audio.
        This attribute is defined in build()
    fold_list : list
        List of the pre-defined fold names.
        e.g. ['fold1', 'fold2', 'fold3', ...]
    label_list : list
        List of class labels.
        e.g. ['dog', 'siren', ...]

    Examples
    --------
    To create a new dataset, it is necessary to define a class that inherits
    from Dataset. Then is required to define the build, generate_file_lists,
    get_annotations and download (if online available) methods.

    >>> from dcase_models.util import list_wav_files

    >>> class TestDataset(Dataset):
    >>>     def __init__(self, dataset_path):
    >>>         super().__init__(dataset_path)

    >>>     def build(self):
    >>>         self.audio_path = os.path.join(self.dataset_path, 'audio')
    >>>         self.fold_list = ["train", "validate", "test"]
    >>>         self.label_list = ["cat", "dog"]

    >>>     def generate_file_lists(self):
    >>>         for fold in self.fold_list:
    >>>             audio_folder = os.path.join(self.audio_path, fold)
    >>>             self.file_lists[fold] = list_wav_files(audio_folder)

    >>>     def get_annotations(self, file_path, features):
    >>>         y = np.zeros((len(features), len(self.label_list)))
    >>>         class_ix = int(os.path.basename(file_path).split('-')[1])
    >>>         y[:, class_ix] = 1
    >>>         return y

    >>>     def download(self, force_download=False):
    >>>         zenodo_url = "https://zenodo.org/record/123456/files"
    >>>         zenodo_files = ["TestData.tar.gz"]
    >>>         downloaded = super().download(
    >>>             zenodo_url, zenodo_files, force_download
    >>>         )
    >>>         if downloaded:
    >>>             move_all_files_to_parent(self.dataset_path, "TestData")
    >>>             self.set_as_downloaded()

    """
    def __init__(self, dataset_path):
        """ Init Dataset

        """

        self.dataset_path = dataset_path
        self.file_lists = {}
        self.build()

    def build(self):
        """ Builds the dataset.

        Define specific attributes of the dataset.
        It's mandatory to define audio_path, fold_list and label_list.
        Other attributes may be defined here (url, authors, etc.).

        """

        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.fold_list = ['fold1', 'fold2', 'fold3']
        self.label_list = ['class1', 'class2', 'class3']

    def generate_file_lists(self):
        """ Creates file_lists, a dict that includes a list of files per fold.

        Each dataset has a different way of organizing the files. This
        function defines the dataset structure.
        """

        # Creates an empty dict
        self.file_lists = {fold: [] for fold in self.fold_list}

    def get_annotations(self, file_path, features, time_resolution):
        """ Returns the annotations of the file in file_path.

        Parameters
        ----------
        file_path : str
            Path to the file
        features : ndarray
            nD array with the features of file_path
        time_resolution : float
            Time resolution of the features

        Returns
        -------
        ndarray
            Annotations of the file file_path
            Expected output shape: (features.shape[0], len(self.label_list))

        """
        raise NotImplementedError

    def download(self, zenodo_url, zenodo_files, force_download=False):
        """ Downloads and decompresses the dataset from zenodo.

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
        """ Saves a download.txt file in dataset_path as a downloaded flag.

        """
        log_file = os.path.join(self.dataset_path, 'download.txt')
        with open(log_file, 'w') as txt_file:
            # pass
            txt_file.write('')

    def check_if_downloaded(self):
        """ Checks if the dataset was downloaded.

        Just checks if exists download.txt file.

        Further checks in the future.

        """
        log_file = os.path.join(self.dataset_path, 'download.txt')
        return os.path.exists(log_file)

    def get_audio_paths(self, sr=None):
        """ Returns paths to the audio folder.

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
            subfolders = [os.path.join(self.audio_path, 'original')]
        else:
            audio_path = self.audio_path + str(sr)
            subfolders = [os.path.join(audio_path, 'original')]

        return audio_path, subfolders

    def change_sampling_rate(self, new_sr):
        """ Changes the sampling rate of each wav file in audio_path.

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

        for path_to_file in progressbar(list_wav_files(self.audio_path)):
            path_to_destination = path_to_file.replace(
                self.audio_path, new_audio_folder
            )
            if os.path.exists(path_to_destination):
                continue
            tfm.build(path_to_file, path_to_destination)

    def check_sampling_rate(self, sr):
        """ Checks if dataset was resampled before.

        For now, only checks if the folder {audio_path}{sr} exists and
        each wav file present in audio_path is also present in
        {audio_path}{sr}.

        Parameters
        ----------
        sr : int
            Sampling rate.

        Returns
        -------
        bool
            True if the dataset was resampled before.

        """

        audio_path, subfolders = self.get_audio_paths(sr)
        audio_folder_sr = subfolders[0]
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

    def convert_to_wav(self, remove_original=False):
        """ Converts each file in the dataset to wav format.

        If remove_original is False, the original files will be deleted

        Parameters
        ----------
        remove_original : bool
            Remove original files.

        """
        tfm = sox.Transformer()

        for path_to_file in list_all_files(self.audio_path):
            if path_to_file.endswith('wav'):
                continue
            path_to_destination = path_to_file.replace(
                os.path.splitext(path_to_file)[1], '.wav'
            )
            if os.path.exists(path_to_destination):
                continue

            tfm.build(path_to_file, path_to_destination)

            if remove_original:
                os.remove(path_to_file)
