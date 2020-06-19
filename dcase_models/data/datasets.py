import glob
import os
import numpy as np
import sys
import csv
from pandas import read_csv
import yaml
from sed_eval.util.event_roll import event_list_to_event_roll
from librosa.util import fix_length

from .dataset_base import Dataset
from ..utils.files import move_all_files_to_parent, move_all_files_to
from ..utils.files import mkdir_if_not_exists

import inspect


class UrbanSound8k(Dataset):
    ''' UrbanSound8k dataset class '''

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
            self.file_lists[fold] = sorted(
                glob.glob(os.path.join(audio_folder, '*.wav'))
            )

    def get_annotations(self, file_path, features):
        y = np.zeros((len(features), len(self.label_list)))
        class_ix = int(os.path.basename(file_path).split('-')[1])
        y[:, class_ix] = 1
        return y

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/1203745/files"
        zenodo_files = ["UrbanSound8K.tar.gz"]
        resp = super().download(
            zenodo_url, zenodo_files, force_download
        )
        if resp is not None:
            move_all_files_to_parent(self.dataset_path, "UrbanSound8K")
            self.set_as_downloaded()


class ESC50(Dataset):
    ''' ESC50 dataset class '''

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
            all_files = sorted(
                glob.glob(os.path.join(self.audio_path, '*.wav')))
            for fil in all_files:
                basename = self.get_basename_wav(fil)
                if basename in self.metadata:
                    if self.metadata[basename]['fold'] == fold:
                        self.file_lists[fold].append(fil)

    def get_annotations(self, file_name, features):
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
        super().download(
            github_url, github_files, force_download
        )
        move_all_files_to_parent(self.dataset_path, "ESC-50-master")
        self.set_as_downloaded()


class ESC10(ESC50):
    ''' ESC10 dataset class '''

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
    ''' URBAN-SED dataset class '''

    def __init__(self, dataset_path,
                 sequence_time=1.0, sequence_hop_time=0.5,
                 metric_resolution_sec=1.0):
        self.sequence_time = sequence_time
        self.sequence_hop_time = sequence_hop_time
        self.metric_resolution_sec = metric_resolution_sec

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
        super().generate_file_lists()
        self.wav_to_labels = {}
        for fold in self.fold_list:
            for fil in self.file_lists[fold]:
                label_file = os.path.basename(fil).split('.')[0] + '.txt'
                self.wav_to_labels[fil] = os.path.join(
                    self.annotations_folder, fold, label_file)

    def get_annotations(self, file_name, features, time_resolution=1.0):
        label_file = self.wav_to_labels[file_name]
        labels = read_csv(label_file, delimiter='\t', header=None)
        labels.columns = ['event_onset', 'event_offset', 'event_label']
        event_roll = event_list_to_event_roll(
            labels.to_dict('records'),
            self.label_list,
            self.sequence_hop_time
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

        super().download(
            zenodo_url, zenodo_files, force_download
        )
        move_all_files_to_parent(self.dataset_path, "URBAN-SED_v2.0.0")
        self.set_as_downloaded()


class SONYC_UST(Dataset):
    ''' SONYC-UST dataset class '''

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
        filename_to_split = self.metadata[[
            'audio_filename', 'split']].drop_duplicates()
        all_files_in_metadata = filename_to_split['audio_filename'].to_list()
        splits = filename_to_split['split'].to_list()

        self.file_lists = {}
        for fold in self.fold_list:
            self.file_lists[fold] = []
            all_files = sorted(
                glob.glob(os.path.join(self.audio_folder, '*.wav')))
            assert len(all_files) != 0
            for fil in all_files:
                basename = os.path.basename(fil)
                if basename in all_files_in_metadata:
                    j = all_files_in_metadata.index(basename)
                    if splits[j] == fold:
                        self.file_lists[fold].append(fil)

    def get_annotations(self, file_name, features):
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
            # class present if any annotator check presence
            y[class_ix-1] = np.sum(metadata_of_file[class_column].values) >= 1

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
    ''' Base class for TAUUrbanAcousticScenes datasets'''

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

    def get_annotations(self, file_name, features):
        y = np.zeros((len(features), len(self.label_list)))
        basename = os.path.basename(file_name)
        # delete file extension
        basename = basename.split('.')[0]
        scene_label, city, location_id, segment_id, device_id = basename.split(
            '-')
        class_ix = self.label_list.index(scene_label)
        y[:, class_ix] = 1
        return y

    def download(self, force_download=False):
        pass


class TAUUrbanAcousticScenes2019(_TAUUrbanAcousticScenes):
    ''' TAUUrbanAcousticScenes2019 dataset class '''

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

        super().download(
            zenodo_url, zenodo_files, force_download
        )
        move_all_files_to_parent(
            self.dataset_path, "TAU-urban-acoustic-scenes-2019-development")
        self.set_as_downloaded()


class TAUUrbanAcousticScenes2020Mobile(_TAUUrbanAcousticScenes):
    ''' TAUUrbanAcousticScenes2020Mobile dataset class '''

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

        super().download(
            zenodo_url, zenodo_files, force_download
        )
        move_all_files_to_parent(
            self.dataset_path,
            "TAU-urban-acoustic-scenes-2020-mobile-development")
        self.set_as_downloaded()


class TUTSoundEvents2017(Dataset):
    ''' TUTSoundEvents2017 dataset class '''

    def __init__(self, dataset_path,
                 sequence_time=1.0, sequence_hop_time=0.5,
                 metric_resolution_sec=1.0):
        self.sequence_time = sequence_time
        self.sequence_hop_time = sequence_hop_time
        self.metric_resolution_sec = metric_resolution_sec

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

    def get_annotations(self, file_name, features):
        label_file = self.wav_to_labels[file_name]
        labels = read_csv(label_file, delimiter='\t', header=None)
        labels.columns = ['file_path', 'scene', 'event_onset',
                          'event_offset', 'event_label',
                          'mixture', 'file_name']
        event_roll = event_list_to_event_roll(
            labels.to_dict('records'), self.label_list, self.sequence_hop_time
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
        super().download(
            zenodo_url, zenodo_files, force_download
        )

        move_all_files_to_parent(
            self.dataset_path,
            "TUT-sound-events-2017-development")

        zenodo_url = "https://zenodo.org/record/1040179/files"

        zenodo_files = [
            'TUT-sound-events-2017-evaluation.audio.zip',
            'TUT-sound-events-2017-evaluation.meta.zip',
        ]
        super().download(
            zenodo_url, zenodo_files, force_download
        )
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
    ''' FSDKaggle2018 dataset class '''

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        self.fold_list = ["train", "test"]
        self.meta_path = os.path.join(self.dataset_path, 'meta')
        self.label_list = []

        meta_file_train = os.path.join(
            self.meta_path, 'train_post_competition.csv'
        )
        meta_file_test = os.path.join(
            self.meta_path, 'test_post_competition_scoring_clips.csv'
        )

        self.metadata = {}
        for fold_ix, meta_file in enumerate([meta_file_train, meta_file_test]):
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

                    self.metadata[filename] = {
                        'label': label, 'usage': usage,
                        'freesound_id': freesound_id, 'license': license,
                        'fold': self.fold_list[fold_ix]}
                    if label not in self.label_list:
                        self.label_list.append(label)

        self.label_list.sort()

    def generate_file_lists(self):
        self.file_lists = {fold: [] for fold in self.fold_list}
        for filename in self.metadata.keys():
            fold = self.metadata[filename]['fold']
            file_path = os.path.join(
                self.audio_path, fold, filename
            )
            self.file_lists[fold].append(file_path)

    def get_annotations(self, file_name, features):
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


def get_available_datasets():
    availabe_datasets = {m[0]: m[1] for m in inspect.getmembers(
        sys.modules[__name__], inspect.isclass)
            if m[1].__module__ == __name__ and m[0][0] != '_'}

    return availabe_datasets
