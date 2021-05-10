import os
import numpy as np
import sys
import csv
from pandas import read_csv
import yaml
from sed_eval.util.event_roll import event_list_to_event_roll
from librosa.util import fix_length

from dcase_models.data.dataset_base import Dataset
from dcase_models.util.files import move_all_files_to_parent, move_all_files_to
from dcase_models.util.files import mkdir_if_not_exists, list_wav_files

import inspect


__all__ = ['UrbanSound8k', 'ESC50', 'ESC10', 'URBAN_SED',
           'SONYC_UST', 'TAUUrbanAcousticScenes2019',
           'TAUUrbanAcousticScenes2020Mobile',
           'TUTSoundEvents2017', 'FSDKaggle2018', 'MAVD']


class UrbanSound8k(Dataset):
    """ UrbanSound8k dataset.

    This class inherits all functionality from Dataset and
    defines specific attributs and methods for UrbanSound8k.

    Url: https://urbansounddataset.weebly.com/urbansound8k.html

    J. Salamon,  C.  Jacoby,  and  J.  P.  Bello
    “A  dataset  and  taxonomy  for  urban  sound  research,”
    22st  ACM  International  Conference  on  Multimedia (ACM-MM’14)
    Orlando, FL, USA, November 2014

    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/UrbanSound8k).

    Examples
    --------
    To work with UrbanSound8k dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import UrbanSound8k
    >>> dataset = UrbanSound8k('../datasets/UrbanSound8K')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')

        self.fold_list = ["fold1", "fold2", "fold3", "fold4",
                          "fold5", "fold6", "fold7", "fold8",
                          "fold9", "fold10"]
        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]

    def generate_file_lists(self):
        for fold in self.fold_list:
            audio_folder = os.path.join(self.audio_path, fold)
            self.file_lists[fold] = list_wav_files(audio_folder)

    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        class_ix = int(os.path.basename(file_name).split('-')[1])
        y[:, class_ix] = 1
        return y

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/1203745/files"
        zenodo_files = ["UrbanSound8K.tar.gz"]
        downloaded = super().download(
            zenodo_url, zenodo_files, force_download
        )
        if downloaded:
            move_all_files_to_parent(self.dataset_path, "UrbanSound8K")
            self.set_as_downloaded()


class ESC50(Dataset):
    """ ESC-50 dataset.

    This class inherits all functionality from Dataset and
    defines specific attributes and methods for ESC-50.

    Url: https://github.com/karolpiczak/ESC-50

    K. J. Piczak
    “Esc:  Dataset for environmental sound classification,”
    Proceedings of the 23rd ACM international conference on Multimedia
    Brisbane, Australia, October, 2015.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/ESC50).

    Examples
    --------
    To work with ESC50 dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import ESC50
    >>> dataset = ESC50('../datasets/ESC50')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        # load metadata information and create label_list
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        meta_file = os.path.join(self.dataset_path, 'meta/esc50.csv')
        self.metadata = {}
        if self.check_if_downloaded():
            n_classes = 50
            self.label_list = ['']*n_classes
            with open(meta_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                        continue
                    filename = row[0]
                    fold = 'fold'+row[1]
                    class_ix = int(row[2])
                    class_name = row[3]
                    esc10 = row[4] == 'True'
                    self.metadata[filename] = {
                        'fold': fold, 'class_ix': class_ix,
                        'class_name': class_name, 'esc10': esc10}
                    if class_name not in self.label_list:
                        self.label_list[class_ix] = class_name

        self.fold_list = ["fold1", "fold2", "fold3", "fold4", "fold5"]
        self.evaluation_mode = 'cross-validation'

    def generate_file_lists(self):
        self.file_lists = {}
        for fold in self.fold_list:
            self.file_lists[fold] = []
            # all_files = sorted(
            #    glob.glob(os.path.join(self.audio_path, '*.wav')))
            all_files = list_wav_files(self.audio_path)
            for fil in all_files:
                basename = self.get_basename_wav(fil)
                if basename in self.metadata:
                    if self.metadata[basename]['fold'] == fold:
                        self.file_lists[fold].append(fil)

    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(self.label_list)))
        basename = self.get_basename_wav(file_name)
        class_ix = self.metadata[basename]['class_ix']
        y[class_ix] = 1
        y = np.expand_dims(y, 0)
        y = np.repeat(y, len(features), 0)
        return y

    def get_basename_wav(self, filename):
        # convert ..../xxxx.npy in xxxx.wav
        return os.path.basename(filename).split('.')[0] + '.wav'

    def download(self, force_download=False):
        github_url = "https://github.com/karoldvl/ESC-50/archive/"
        github_files = ["master.zip"]
        downloaded = super().download(
            github_url, github_files, force_download
        )
        if downloaded:
            move_all_files_to_parent(self.dataset_path, "ESC-50-master")
            self.set_as_downloaded()


class ESC10(ESC50):
    """ ESC-10 dataset.

    This class inherits all functionality from Dataset and
    defines specific attributes and methods for ESC-10.

    ESC-10 is a subsampled version of ESC-50.

    Url: https://github.com/karolpiczak/ESC-50

    K. J. Piczak
    “Esc:  Dataset for environmental sound classification,”
    Proceedings of the 23rd ACM international conference on Multimedia
    Brisbane, Australia, October, 2015.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/ESC50).

    Examples
    --------
    To work with ESC10 dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import ESC10
    >>> dataset = ESC10('../datasets/ESC50')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        super().build()

        # then change self.metadata and self.label_list to keep only ESC-10
        new_metada = {}
        new_label_list_ids = []
        for j in self.metadata.keys():
            if self.metadata[j]['esc10']:
                new_metada[j] = self.metadata[j].copy()
                if new_metada[j]['class_ix'] not in new_label_list_ids:
                    new_label_list_ids.append(new_metada[j]['class_ix'])

        new_label_list_ids.sort()
        new_label_list = []
        new_label_list = [self.label_list[i] for i in new_label_list_ids]

        self.metadata = new_metada.copy()
        self.label_list = new_label_list.copy()
        for j in self.metadata.keys():
            assert self.metadata[j]['esc10']
            self.metadata[j]['class_ix'] = [i for i, x in enumerate(
                self.label_list) if x == self.metadata[j]['class_name']][0]

        # regenerate self.file_lists
        self.generate_file_lists()


class URBAN_SED(Dataset):
    """ URBAN-SED dataset.

    This class inherits all functionality from Dataset and
    defines specific attributes and methods for URBAN-SED.

    Url: http://urbansed.weebly.com/

    J. Salamon,  D. MacConnell,  M. Cartwright,  P. Li,  and J. P.Bello.
    "Scaper: A library for soundscape synthesis and augmentation".
    IEEE Workshop on Applications of Signal Processing to Audio and Acoustics
    New York, USA, October 2017.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/URBAN_SED).

    Examples
    --------
    To work with URBAN_SED dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import URBAN_SED
    >>> dataset = URBAN_SED('../datasets/URBAN_SED')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.annotations_folder = os.path.join(
            self.dataset_path, 'annotations')
        self.fold_list = ["train", "validate", "test"]
        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]
        self.evaluation_mode = 'train-validate-test'

    def generate_file_lists(self):
        for fold in self.fold_list:
            audio_folder = os.path.join(self.audio_path, fold)
            self.file_lists[fold] = list_wav_files(audio_folder)

        self.wav_to_labels = {}
        for fold in self.fold_list:
            for fil in self.file_lists[fold]:
                label_file = os.path.basename(fil).split('.')[0] + '.txt'
                self.wav_to_labels[fil] = os.path.join(
                    self.annotations_folder, fold, label_file)

    def get_annotations(self, file_name, features, time_resolution):
        label_file = self.wav_to_labels[file_name]
        labels = read_csv(label_file, delimiter='\t', header=None)
        labels.columns = ['event_onset', 'event_offset', 'event_label']
        event_roll = event_list_to_event_roll(
            labels.to_dict('records'),
            self.label_list,
            time_resolution
        )
        if event_roll.shape[0] > features.shape[0]:
            event_roll = event_roll[:len(features)]
        else:
            event_roll = fix_length(event_roll, features.shape[0], axis=0)
        assert event_roll.shape[0] == features.shape[0]
        return event_roll

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/1324404/files"
        zenodo_files = ["URBAN-SED_v2.0.0.tar.gz"]

        downloaded = super().download(
            zenodo_url, zenodo_files, force_download
        )
        if downloaded:
            move_all_files_to_parent(self.dataset_path, "URBAN-SED_v2.0.0")
            self.set_as_downloaded()


class SONYC_UST(Dataset):
    """ SONYC-UST dataset.

    This class inherits all functionality from Dataset and
    defines specific attributes and methods for SONYC-UST.

    Version: 2.1.0

    Url: https://zenodo.org/record/3693077

    M. Cartwright, et al.
    "SONYC Urban Sound Tagging (SONYC-UST): A Multilabel Dataset
    from an Urban Acoustic Sensor Network".
    Proceedings of the Workshop on Detection and Classification
    of Acoustic Scenes and Events (DCASE), 2019.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/SONYC_UST).

    Examples
    --------
    To work with SONYC_UST dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import SONYC_UST
    >>> dataset = SONYC_UST('../datasets/SONYC_UST')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.fold_list = ["train", "validate"]
        self.evaluation_mode = 'train-validate-test'
        self.meta_file = os.path.join(self.dataset_path, 'annotations.csv')
        self.taxonomy_file = os.path.join(
            self.dataset_path, 'dcase-ust-taxonomy.yaml')

        self.metada = {}
        self.label_list = []
        if self.check_if_downloaded():
            self.metadata = read_csv(self.meta_file).sort_values(
                'audio_filename')
            with open(self.taxonomy_file, 'r') as f:
                self.label_list = yaml.load(f, Loader=yaml.Loader)

    def generate_file_lists(self):
        self.file_lists = {}
        all_files = list_wav_files(self.audio_path)
        assert len(all_files) != 0
        for fold in self.fold_list:
            if fold == 'train':
                metadata_fold = self.metadata[self.metadata['split'] == fold]
            else:
                metadata_fold = self.metadata[
                    ((self.metadata['split'] == fold) &
                     (self.metadata['annotator_id'] == 0))
                ]
            filename_list_fold = metadata_fold[
                'audio_filename'].drop_duplicates().to_list()
            self.file_lists[fold] = []
            for fil in all_files:
                basename = os.path.basename(fil)
                if basename in filename_list_fold:
                    self.file_lists[fold].append(fil)

    def get_annotations(self, file_name, features, time_resolution):
        # only coarse level
        # TODO add fine level
        n_classes_coarse_level = len(self.label_list['coarse'])
        y = np.zeros(n_classes_coarse_level)
        basename = os.path.basename(file_name).split('.')[0] + '.wav'

        metadata_of_file = self.metadata[
            self.metadata['audio_filename'] == basename]
        for class_ix in self.label_list['coarse']:
            class_column = str(class_ix) + '_' + \
                self.label_list['coarse'][class_ix] + '_presence'

            if metadata_of_file['split'].values[0] == 'train':
                # class present if any annotator check presence
                y[class_ix-1] = np.sum(
                    metadata_of_file[class_column].values) >= 1
            else:
                # class present if annotator 0 check presence
                if 0 in metadata_of_file['annotator_id'].values:
                    ix = np.argwhere(
                        metadata_of_file['annotator_id'].values == 0)
                    y[class_ix-1] = metadata_of_file[
                        class_column].values[ix] >= 1

        y = np.expand_dims(y, 0)
        y = np.repeat(y, len(features), 0)
        return y

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/3693077/files"
        zenodo_files = ["annotations.csv", "audio.tar.gz",
                        "dcase-ust-taxonomy.yaml", "README.md"]
        super().download(
            zenodo_url, zenodo_files, force_download
        )
        self.set_as_downloaded()


class _TAUUrbanAcousticScenes(Dataset):
    """ Base class for TAU Urban Acoustic Scenes datasets.

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.fold_list = ["train", "test"]
        self.meta_file = os.path.join(self.dataset_path, 'meta.csv')
        self.label_list = ['airport', 'shopping_mall', 'metro_station',
                           'street_pedestrian', 'public_square',
                           'street_traffic', 'tram', 'bus', 'metro', 'park']

        self.evaluation_setup_train = os.path.join(
            self.dataset_path, 'evaluation_setup', 'fold1_train.csv')
        self.evaluation_setup_test = os.path.join(
            self.dataset_path, 'evaluation_setup', 'fold1_test.csv')
        self.annotations_folder = os.path.join(
            self.dataset_path, 'annotations')

    def generate_file_lists(self):
        self.file_lists = {}
        evaluation_files = [self.evaluation_setup_train,
                            self.evaluation_setup_test]
        for j, fold in enumerate(['train', 'test']):
            self.file_lists[fold] = []
            csv_filename = evaluation_files[j]
            with open(csv_filename) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                        continue
                    file_name = row[0].split('/')[-1]
                    self.file_lists[fold].append(
                        os.path.join(self.audio_path, file_name)
                    )

    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        basename = os.path.basename(file_name)
        # delete file extension
        basename = basename.split('.')[0]
        scene_label, city, location_id, segment_id, device_id = basename.split(
            '-')
        class_ix = self.label_list.index(scene_label)
        y[:, class_ix] = 1
        return y

    def download(self, zenodo_url, zenodo_files, force_download=False):
        return super().download(zenodo_url, zenodo_files,
                                force_download=force_download)


class TAUUrbanAcousticScenes2019(_TAUUrbanAcousticScenes):
    """ TAU Urban Acoustic Scenes 2019 dataset.

    This class inherits all functionality from Dataset and
    defines specific attributes and methods for TAU Urban
    Acoustic Scenes 2019.

    Url: https://zenodo.org/record/2589280

    A.  Mesaros,  T.  Heittola,  and  T.  Virtanen.
    "A  multi-devicedataset for urban acoustic scene classification".
    Proceedings of  the  Detection  and  Classification  of  Acoustic
    Scenes and Events 2018 Workshop (DCASE 2018).
    November 2018.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/TAUUrbanAcousticScenes2019).

    Examples
    --------
    To work with TAUUrbanAcousticScenes2019 dataset, just initialize this
    class with the path to the dataset.

    >>> from dcase_models.data.datasets import TAUUrbanAcousticScenes2019
    >>> dataset = TAUUrbanAcousticScenes2019(
        '../datasets/TAUUrbanAcousticScenes2019')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/2589280/files"
        zenodo_files = [
            "TAU-urban-acoustic-scenes-2019-development.audio.%d.zip" %
            j for j in range(1, 22)]
        zenodo_files.append(
            'TAU-urban-acoustic-scenes-2019-development.doc.zip')
        zenodo_files.append(
            'TAU-urban-acoustic-scenes-2019-development.meta.zip')

        downloaded = super().download(
            zenodo_url, zenodo_files, force_download
        )
        if downloaded:
            move_all_files_to_parent(
                self.dataset_path,
                "TAU-urban-acoustic-scenes-2019-development")
            self.set_as_downloaded()


class TAUUrbanAcousticScenes2020Mobile(_TAUUrbanAcousticScenes):
    """ TAU Urban Acoustic Scenes 2019 dataset.

    This class inherits all functionality from Dataset and
    defines specific attributes and methods for TAU Urban
    Acoustic Scenes 2020 Mobile.

    Url: https://zenodo.org/record/3819968

    T.  Heittola,  A.  Mesaros,  and  T.  Virtanen.
    "Acoustic  sceneclassification   in   DCASE   2020  challenge:
    generalizationacross devices and low complexity solutions".
    Proceedings of  the  Detection  and  Classification  of  Acoustic
    Scenes and Events  2020  Workshop  (DCASE  2020).
    2020

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/TAUUrbanAcousticScenes2020Mobile).

    Examples
    --------
    To work with TAUUrbanAcousticScenes2020Mobile dataset, just initialize this
    class with the path to the dataset.

    >>> from dcase_models.data.datasets import TAUUrbanAcousticScenes2020Mobile
    >>> dataset = TAUUrbanAcousticScenes2020Mobile(
        '../datasets/TAUUrbanAcousticScenes2020Mobile')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/3819968/files"
        zenodo_files = [
            "TAU-urban-acoustic-scenes-2020-mobile-development.audio.%d.zip" %
            j for j in range(1, 17)]
        zenodo_files.append(
            'TAU-urban-acoustic-scenes-2020-mobile-development.doc.zip')
        zenodo_files.append(
            'TAU-urban-acoustic-scenes-2020-mobile-development.meta.zip')

        downloaded = super().download(
            zenodo_url, zenodo_files, force_download
        )
        print(downloaded)
        if downloaded:
            move_all_files_to_parent(
                self.dataset_path,
                "TAU-urban-acoustic-scenes-2020-mobile-development")
            self.set_as_downloaded()


class TUTSoundEvents2017(Dataset):
    """ TUT Sound Events 2017 dataset.

    This class inherits all functionality from Dataset and
    defines specific attributes and methods for TUT Sound
    Events 2017.

    Url: https://zenodo.org/record/814831

    A. Mesaros et al.
    DCASE 2017 challenge setup: tasks, datasets and baseline system.
    Detection and Classification of Acoustic Scenes and Events 2017
    Workshop (DCASE2017), 85–92.
    November 2017.


    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/TUTSoundEvents2017).

    Examples
    --------
    To work with TUTSoundEvents2017 dataset, just initialize this
    class with the path to the dataset.

    >>> from dcase_models.data.datasets import TUTSoundEvents2017
    >>> dataset = TUTSoundEvents2017('../datasets/TUTSoundEvents2017')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.fold_list = ["fold1", "fold2", "fold3", "fold4"]
        self.meta_path = os.path.join(self.dataset_path, 'meta')
        self.label_list = ['brakes squeaking', 'car', 'children',
                           'large vehicle', 'people speaking',
                           'people walking']

        self.evaluation_setup_path = os.path.join(
            self.dataset_path, 'evaluation_setup'
        )

    def generate_file_lists(self):
        self.file_lists = {}
        self.wav_to_labels = {}
        for j, fold in enumerate(self.fold_list):
            self.file_lists[fold] = []
            evaluation_setup_file = os.path.join(
                self.evaluation_setup_path, 'street_%s_test.txt' % fold
            )
            with open(evaluation_setup_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                for row in csv_reader:
                    file_name = row[0].split('/')[-1]
                    file_path = os.path.join(
                        self.audio_path, 'street', file_name
                    )
                    self.file_lists[fold].append(file_path)

                    file_ann = file_path.replace(
                        self.audio_path, self.meta_path
                    )
                    file_ann = file_ann.replace('.wav', '.ann')
                    self.wav_to_labels[file_path] = file_ann

        # test folder
        self.file_lists['test'] = []
        evaluation_setup_file = os.path.join(
            self.evaluation_setup_path, 'street_test.txt'
        )
        with open(evaluation_setup_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                file_name = row[0].split('/')[-1]
                file_path = os.path.join(self.audio_path, 'street', file_name)
                self.file_lists['test'].append(file_path)

                file_ann = file_path.replace(self.audio_path, self.meta_path)
                file_ann = file_ann.replace('.wav', '.ann')
                self.wav_to_labels[file_path] = file_ann

    def get_annotations(self, file_name, features, time_resolution):
        label_file = self.wav_to_labels[file_name]
        labels = read_csv(label_file, delimiter='\t', header=None)

        if labels.shape[1] == 3:
            labels.columns = ['event_onset', 'event_offset', 'event_label']
        else:
            labels.columns = ['file_path', 'scene', 'event_onset',
                              'event_offset', 'event_label',
                              'mixture', 'file_name']
        event_roll = event_list_to_event_roll(
            labels.to_dict('records'), self.label_list, time_resolution
        )
        if event_roll.shape[0] > features.shape[0]:
            event_roll = event_roll[:len(features)]
        else:
            event_roll = fix_length(event_roll, features.shape[0], axis=0)
        assert event_roll.shape[0] == features.shape[0]
        return event_roll

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/814831/files"

        zenodo_files = [
            'TUT-sound-events-2017-development.audio.1.zip',
            'TUT-sound-events-2017-development.audio.2.zip',
            'TUT-sound-events-2017-development.doc.zip',
            'TUT-sound-events-2017-development.meta.zip'
        ]
        downloaded = super().download(
            zenodo_url, zenodo_files, force_download
        )
        if downloaded:
            move_all_files_to_parent(
                self.dataset_path,
                "TUT-sound-events-2017-development")

        zenodo_url = "https://zenodo.org/record/1040179/files"

        zenodo_files = [
            'TUT-sound-events-2017-evaluation.audio.zip',
            'TUT-sound-events-2017-evaluation.meta.zip',
        ]
        downloaded = super().download(
            zenodo_url, zenodo_files, force_download
        )
        if downloaded:
            move_all_files_to(
                os.path.join(
                    self.dataset_path,
                    "TUT-sound-events-2017-evaluation/audio/street"
                ),
                os.path.join(self.dataset_path, "audio/street")
            )
            move_all_files_to(
                os.path.join(
                    self.dataset_path,
                    "TUT-sound-events-2017-evaluation/meta/street"
                ),
                os.path.join(self.dataset_path, "meta/street")
            )
            move_all_files_to(
                os.path.join(
                    self.dataset_path,
                    "TUT-sound-events-2017-evaluation/evaluation_setup"
                ),
                os.path.join(self.dataset_path, "evaluation_setup")
            )
            self.set_as_downloaded()


class FSDKaggle2018(Dataset):
    """ FSDKaggle2018 dataset.

    This class inherits all functionality from Dataset and
    defines specific attributes and methods for FSDKaggle2018.

    Url: https://zenodo.org/record/2552860

    Eduardo Fonseca et al.
    "General-purpose Tagging of Freesound Audio with AudioSet Labels:
    Task Description, Dataset, and Baseline".
    Proceedings of the DCASE 2018 Workshop.
    2018.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/FSDKaggle2018).

    Examples
    --------
    To work with FSDKaggle2018 dataset, just initialize this
    class with the path to the dataset.

    >>> from dcase_models.data.datasets import FSDKaggle2018
    >>> dataset = FSDKaggle2018('../datasets/FSDKaggle2018')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.fold_list = ["train", "validate", "test"]
        self.meta_path = os.path.join(self.dataset_path, 'meta')
        self.label_list = []

        meta_file_train = os.path.join(
            self.meta_path, 'train_post_competition.csv'
        )
        meta_file_test = os.path.join(
            self.meta_path, 'test_post_competition_scoring_clips.csv'
        )

        self.metadata = {}
        if self.check_if_downloaded():
            for meta_file in [meta_file_train, meta_file_test]:
                with open(meta_file) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 0:
                            line_count += 1
                            continue
                        filename = row[0]
                        label = row[1]
                        usage = row[2]
                        freesound_id = row[3]
                        license = row[4]

                        if meta_file == meta_file_train:
                            fold = 'train'
                        else:
                            if usage == 'Public':
                                fold = 'validate'
                            else:
                                fold = 'test'

                        self.metadata[filename] = {
                            'label': label, 'usage': usage,
                            'freesound_id': freesound_id, 'license': license,
                            'fold': fold}
                        if label not in self.label_list:
                            self.label_list.append(label)

            self.label_list.sort()

    def generate_file_lists(self):
        self.file_lists = {fold: [] for fold in self.fold_list}
        for filename in self.metadata.keys():
            fold = self.metadata[filename]['fold']
            fold_folder = fold
            if fold == 'validate':
                fold_folder = 'test'
            file_path = os.path.join(
                self.audio_path, fold_folder, filename
            )
            self.file_lists[fold].append(file_path)

    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        label_name = self.metadata[os.path.basename(file_name)]['label']
        label_index = self.label_list.index(label_name)
        y[:, label_index] = 1
        return y

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/2552860/files"

        zenodo_files = [
            'FSDKaggle2018.audio_test.zip',
            'FSDKaggle2018.audio_train.zip',
            'FSDKaggle2018.doc.zip',
            'FSDKaggle2018.meta.zip'
        ]
        super().download(
            zenodo_url, zenodo_files, force_download
        )

        mkdir_if_not_exists(self.audio_path)

        os.rename(
            os.path.join(self.dataset_path, 'FSDKaggle2018.audio_train'),
            os.path.join(self.audio_path, 'train'),
        )
        os.rename(
            os.path.join(self.dataset_path, 'FSDKaggle2018.audio_test'),
            os.path.join(self.audio_path, 'test'),
        )
        os.rename(
            os.path.join(self.dataset_path, 'FSDKaggle2018.meta'),
            os.path.join(self.dataset_path, 'meta'),
        )
        os.rename(
            os.path.join(self.dataset_path, 'FSDKaggle2018.doc'),
            os.path.join(self.dataset_path, 'doc'),
        )

        self.set_as_downloaded()


class MAVD(Dataset):
    """ MAVD-traffic dataset.

    This class inherits all functionality from Dataset and
    defines specific attributes and methods for MAVD-traffic.

    Url: https://zenodo.org/record/3338727

    P. Zinemanas,  P. Cancela,  and  M. Rocamora.
    "MAVD: a dataset for sound event detection in urban environments"
    Proceedings of the Detection and Classification of Acoustic
    Scenes and Events 2019 Workshop (DCASE 2019).
    October, 2019.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/MAVD).

    Examples
    --------
    To work with MAVD dataset, just initialize this
    class with the path to the dataset.

    >>> from dcase_models.data.datasets import MAVD
    >>> dataset = MAVD('../datasets/MAVD')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.annotations_path = os.path.join(self.dataset_path, 'annotations')
        self.fold_list = ["train", "validate", "test"]

        # Only vehicle level for now
        # TODO: Add other levels
        self.vehicle_list = ['car', 'bus', 'truck', 'motorcycle']
        self.component_list = ['engine_idling', 'engine_accelerating',
                               'brakes', 'wheel_rolling', 'compressor']
        self.label_list = self.vehicle_list + self.component_list

    def generate_file_lists(self):
        for fold in self.fold_list:
            audio_folder = os.path.join(self.audio_path, fold)
            self.file_lists[fold] = list_wav_files(audio_folder)

    def get_annotations(self, file_name, features, time_resolution):
        audio_path, _ = self.get_audio_paths()
        label_file = file_name.replace(
            audio_path,
            self.annotations_path
        ).replace('.wav', '.txt')
        labels = read_csv(label_file, delimiter='\t', header=None)
        labels.columns = ['event_onset', 'event_offset', 'event_label']
        labels_dict = labels.to_dict('records')

        event_roll = np.zeros((len(features), len(self.label_list)))
        for event in labels_dict:
            event_label = event['event_label']
            for sub_label in event_label.split('/'):
                if sub_label in self.label_list:
                    label_ix = self.label_list.index(sub_label)

                    event_onset = event['event_onset']
                    event_offset = event['event_offset']

                    onset = int(np.floor(
                        event_onset / float(time_resolution))
                    )
                    offset = int(np.ceil(
                        event_offset / float(time_resolution))
                    )

                    event_roll[onset:offset, label_ix] = 1

        return event_roll

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/3338727/files/"

        zenodo_files = ['audio_train.zip', 'audio_validate.zip',
                        'audio_test.zip', 'annotations_train.zip',
                        'annotations_validate.zip',
                        'annotations_test.zip', 'README']

        super().download(
            zenodo_url, zenodo_files, force_download
        )

        mkdir_if_not_exists(self.audio_path)
        mkdir_if_not_exists(self.annotations_path)
        for fold in self.fold_list:
            os.rename(
                os.path.join(self.dataset_path, 'audio_%s' % fold),
                os.path.join(self.audio_path, fold)
            )
            os.rename(
                os.path.join(self.dataset_path, 'annotations_%s' % fold),
                os.path.join(self.annotations_path, fold)
            )

        # Convert .flac to .wav
        self.convert_to_wav()

        self.set_as_downloaded()


def get_available_datasets():
    availabe_datasets = {m[0]: m[1] for m in inspect.getmembers(
        sys.modules[__name__], inspect.isclass)
            if m[1].__module__ == __name__ and m[0][0] != '_'}

    return availabe_datasets
