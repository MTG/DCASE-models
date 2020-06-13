import glob
import os
import numpy as np
import sys
import csv
from pandas import read_csv
import soundfile as sf
import yaml

from .data_generator import DataGenerator
from ..utils.files import move_all_files_to_parent

import inspect


class UrbanSound8k(DataGenerator):
    ''' UrbanSound8k class (almost copy of DataGenerator) '''

    def __init__(self, dataset_path, feature_extractor=None, **kwargs):
        super().__init__(dataset_path, feature_extractor, **kwargs)

    def download_dataset(self):
        zenodo_url = "https://zenodo.org/record/1203745/files"
        zenodo_files = ["UrbanSound8K.tar.gz"]
        super().download_dataset(zenodo_url, zenodo_files)
        move_all_files_to_parent(self.dataset_path, "UrbanSound8K")
        self.set_dataset_download_finish()


class ESC50(DataGenerator):
    ''' DataGenerator for ESC50 Dataset '''

    def __init__(self, dataset_path, feature_extractor=None, **kwargs):
        
        # load metadata information and create label_list
        meta_file = os.path.join(dataset_path, 'meta/esc50.csv')

        self.metadata = {}
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

        super().__init__(dataset_path, feature_extractor, **kwargs)

    def build(self):
        self.fold_list = ["fold1", "fold2", "fold3", "fold4", "fold5"]
        self.evaluation_mode = 'cross-validation'

        # all wav files are in the same folder
        self.folders_list = [{'audio': os.path.join(self.audio_folder),
                              'features': os.path.join(self.features_folder)}]

    def generate_file_lists(self):
        self.file_lists = {}
        for fold in self.fold_list:
            self.file_lists[fold] = []
            features_folder = os.path.join(self.features_folder, self.features)
            all_files = sorted(
                glob.glob(os.path.join(features_folder, '*.npy')))
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

    def download_dataset(self):
        github_url = "https://github.com/karoldvl/ESC-50/archive/"
        github_files = ["master.zip"]
        super().download_dataset(github_url, github_files)
        move_all_files_to_parent(self.dataset_path, "ESC-50-master")
        self.set_dataset_download_finish()


class ESC10(ESC50):
    ''' DataGenerator for ESC10 Dataset '''

    def __init__(self, dataset_path, feature_extractor=None, **kwargs):
        super().__init__(dataset_path, feature_extractor, **kwargs)

        # then change self.metadata and self.laberl_lsit to keep only ESC-10
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
        print('label_list', new_label_list)

        self.metadata = new_metada.copy()
        self.label_list = new_label_list.copy()
        for j in self.metadata.keys():
            assert self.metadata[j]['esc10']
            self.metadata[j]['class_ix'] = [i for i, x in enumerate(
                self.label_list) if x == self.metadata[j]['class_name']][0]

        # regenerate self.file_lists
        self.generate_file_lists()


class URBAN_SED(DataGenerator):
    ''' DataGenerator for URBAN-SED Dataset '''

    def __init__(self, dataset_path, feature_extractor=None,
                 sequence_time=1.0, sequence_hop_time=0.5,
                 metric_resolution_sec=1.0, **kwargs):
        self.sequence_time = sequence_time
        self.sequence_hop_time = sequence_hop_time
        self.metric_resolution_sec = metric_resolution_sec
        super().__init__(dataset_path, feature_extractor, **kwargs)

    def build(self):
        self.annotations_folder = os.path.join(
            self.dataset_path, 'annotations')
        self.fold_list = ["train", "validate", "test"]
        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]
        self.evaluation_mode = 'train-validate-test'

        self.folders_list = []
        for fold in self.fold_list:
            audio_path = os.path.join(self.audio_folder, fold)
            features_path = os.path.join(self.features_folder, fold)
            audio_features_path = {'audio': audio_path,
                                   'features': features_path}
            self.folders_list.append(audio_features_path)

    def generate_file_lists(self):
        super().generate_file_lists()
        self.features_to_labels = {}
        self.features_to_wav = {}
        for fold in self.fold_list:
            for fil in self.file_lists[fold]:
                label_file = os.path.basename(fil).split('.')[0] + '.txt'
                self.features_to_labels[fil] = os.path.join(
                    self.annotations_folder, fold, label_file)
                audio_file = os.path.basename(fil).split('.')[0] + '.wav'
                self.features_to_wav[fil] = os.path.join(
                    self.audio_folder, fold, audio_file)

    def data_generation(self, list_files_temp):
        features_list = []
        annotations_sequences = []
        annotations_grid_metrics = []
        for file_name in list_files_temp:
            features = np.load(file_name)
            features_list.append(features)
            y_sequences = self.get_annotations(
                file_name, features,
                time_resolution=self.sequence_hop_time)
            y_grid_metrics = self.get_annotations(
                file_name, features,
                time_resolution=self.metric_resolution_sec)
            # print(y_sequences.shape,y_grid_metrics.shape,features.shape)
            y_sequences = y_sequences[:len(features)]
            # print(y_sequences.shape, features.shape)
            assert y_sequences.shape[0] == features.shape[0]
            # print('seq', y_sequences.shape, 'grid', y_grid_metrics.shape)
            annotations_sequences.append(y_sequences)
            annotations_grid_metrics.append(y_grid_metrics)
            # annotations.append({'y_frames': y_frames,
            #                     'y_grid_metrics': y_grid_metrics})

        return features_list, [annotations_sequences, annotations_grid_metrics]

    def get_annotations(self, file_name, features, time_resolution=1.0):
        label_file = self.features_to_labels[file_name]
        audio_file = self.features_to_wav[file_name]
        f = sf.SoundFile(audio_file)
        audio_len_sec = len(f) / f.samplerate
        labels = read_csv(label_file, delimiter='\t', header=None)
        labels.columns = ['event_onset', 'event_offset', 'event_label']

        # print(features.shape[0], self.sequence_hop_time, time_resolution)
        N_seqs = int(
            np.floor((audio_len_sec + self.sequence_time) / time_resolution))
        event_roll = np.zeros((N_seqs, len(self.label_list)))
        # print(event_roll.shape)
        for event in labels.to_dict('records'):
            pos = self.label_list.index(event['event_label'])

            event_onset = event['event_onset']
            event_offset = event['event_offset']

            # event_offset = 5.0
            # event_onset = 0.0

            # math.floor
            onset = int(np.round(event_onset * 1 / float(time_resolution)))
            offset = int(np.round(event_offset * 1 /
                                  float(time_resolution))) + 1  # math.ceil
            # print(event_onset,event_offset,onset,offset)
            event_roll[onset:offset, pos] = 1
        return event_roll

    def get_data_for_training(self, fold_test='test'):
        # train-val-test mode
        X_val = self.data['validate']['X']
        Y_val = self.data['validate']['Y'][1]  # grid time of metrics

        X_train = np.concatenate(self.data['train']['X'], axis=0)
        # grid time of instances for training
        Y_train = np.concatenate(self.data['train']['Y'][0], axis=0)

        return X_train, Y_train, X_val, Y_val

    def get_data_for_testing(self, fold_test='test'):
        # train-val-test mode
        X_test = self.data[fold_test]['X']
        Y_test = self.data[fold_test]['Y'][1]

        return X_test, Y_test

    def download_dataset(self):
        zenodo_url = "https://zenodo.org/record/1324404/files"
        zenodo_files = ["URBAN-SED_v2.0.0.tar.gz"]

        super().download_dataset(zenodo_url, zenodo_files)
        move_all_files_to_parent(self.dataset_path, "URBAN-SED_v2.0.0")
        self.set_dataset_download_finish()


class SONYC_UST(DataGenerator):
    ''' DataGenerator for SONYC-UST Dataset '''

    def __init__(self, dataset_path, feature_extractor=None, **kwargs):
        super().__init__(dataset_path, feature_extractor, **kwargs)

    def build(self):
        self.fold_list = ["train", "validate"]
        self.evaluation_mode = 'train-validate-test'
        self.meta_file = os.path.join(self.dataset_path, 'annotations.csv')
        self.taxonomy_file = os.path.join(
            self.dataset_path, 'dcase-ust-taxonomy.yaml')

        self.metadata = read_csv(self.meta_file).sort_values('audio_filename')

        with open(self.taxonomy_file, 'r') as f:
            self.label_list = yaml.load(f, Loader=yaml.Loader)

        # all wav files are in the same folder
        self.folders_list = [{'audio': os.path.join(self.audio_folder),
                              'features': os.path.join(self.features_folder)}]

    def generate_file_lists(self):
        filename_to_split = self.metadata[[
            'audio_filename', 'split']].drop_duplicates()
        all_files_in_metadata = filename_to_split['audio_filename'].to_list()
        splits = filename_to_split['split'].to_list()

        self.file_lists = {}
        for fold in self.fold_list:
            self.file_lists[fold] = []
            features_folder = os.path.join(self.features_folder, self.features)
            all_files = sorted(
                glob.glob(os.path.join(self.audio_folder, '*.wav')))
            assert len(all_files) != 0
            for fil in all_files:
                basename = os.path.basename(fil)
                if basename in all_files_in_metadata:
                    j = all_files_in_metadata.index(basename)
                    if splits[j] == fold:
                        file_npy = os.path.join(
                            features_folder, basename.split('.')[0] + '.npy')
                        self.file_lists[fold].append(file_npy)

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

    def get_data_for_training(self, fold_test='test'):
        # train-val-test mode
        X_val = self.data['validate']['X']
        Y_val = self.data['validate']['Y']

        X_train = np.concatenate(self.data['train']['X'], axis=0)
        Y_train = np.concatenate(self.data['train']['Y'], axis=0)

        return X_train, Y_train, X_val, Y_val

    def get_data_for_testing(self, fold_test='test'):
        return None, None

    def download_dataset(self):
        zenodo_url = "https://zenodo.org/record/3693077/files"
        zenodo_files = ["annotations.csv", "audio.tar.gz",
                        "dcase-ust-taxonomy.yaml", "README.md"]
        super().download_dataset(zenodo_url, zenodo_files)
        self.set_dataset_download_finish()


class TAUUrbanAcousticScenes2019(DataGenerator):
    ''' DataGenerator for TAUUrbanAcousticScenes2019 Dataset '''

    def __init__(self, dataset_path, feature_extractor=None, **kwargs):
        super().__init__(dataset_path, feature_extractor, **kwargs)

    def build(self):
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

        # all wav files are in the same folder
        self.folders_list = [{'audio': os.path.join(self.audio_folder),
                              'features': os.path.join(self.features_folder)}]

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
                    file_name_npy = os.path.basename(
                        file_name).split('.')[0] + '.npy'
                    self.file_lists[fold].append(os.path.join(
                        self.features_folder, self.features, file_name_npy))

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

    def get_data_for_training(self, fold_test='test'):
        # train-val-test mode

        X_train = np.concatenate(self.data['train']['X'], axis=0)
        Y_train = np.concatenate(self.data['train']['Y'], axis=0)

        # TODO: make a validation set
        X_val = self.data['train']['X']
        Y_val = self.data['train']['Y']

        return X_train, Y_train, X_val, Y_val

    def get_data_for_testing(self, fold_test='test'):
        X_test = self.data['test']['X']
        Y_test = self.data['test']['Y']
        return X_test, Y_test

    def download_dataset(self):
        zenodo_url = "https://zenodo.org/record/2589280/files"
        zenodo_files = [
            "TAU-urban-acoustic-scenes-2019-development.audio.%d.zip" %
            j for j in range(1, 22)]
        zenodo_files.append(
            'TAU-urban-acoustic-scenes-2019-development.doc.zip')
        zenodo_files.append(
            'TAU-urban-acoustic-scenes-2019-development.meta.zip')

        super().download_dataset(zenodo_url, zenodo_files)
        move_all_files_to_parent(
            self.dataset_path, "TAU-urban-acoustic-scenes-2019-development")
        self.set_dataset_download_finish()


class TAUUrbanAcousticScenes2020Mobile(DataGenerator):
    ''' DataGenerator for TAUUrbanAcousticScenes2020Mobile Dataset '''
    
    def __init__(self, dataset_path, feature_extractor=None, **kwargs):
        super().__init__(dataset_path, feature_extractor, **kwargs)

    def build(self):
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

        # all wav files are in the same folder
        self.folders_list = [{'audio': os.path.join(self.audio_folder),
                              'features': os.path.join(self.features_folder)}]

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
                    file_name_npy = os.path.basename(
                        file_name).split('.')[0] + '.npy'
                    self.file_lists[fold].append(os.path.join(
                        self.features_folder, self.features, file_name_npy))

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

    def get_data_for_training(self, fold_test='test'):
        # train-val-test mode

        X_train = np.concatenate(self.data['train']['X'], axis=0)
        Y_train = np.concatenate(self.data['train']['Y'], axis=0)

        # TODO: make a validation set
        X_val = self.data['train']['X']
        Y_val = self.data['train']['Y']

        return X_train, Y_train, X_val, Y_val

    def get_data_for_testing(self, fold_test='test'):
        X_test = self.data['test']['X']
        Y_test = self.data['test']['Y']
        return X_test, Y_test

    def download_dataset(self):
        zenodo_url = "https://zenodo.org/record/3819968/files"
        zenodo_files = [
            "TAU-urban-acoustic-scenes-2020-mobile-development.audio.%d.zip" %
            j for j in range(1, 17)]
        zenodo_files.append(
            'TAU-urban-acoustic-scenes-2020-mobile-development.doc.zip')
        zenodo_files.append(
            'TAU-urban-acoustic-scenes-2020-mobile-development.meta.zip')

        super().download_dataset(zenodo_url, zenodo_files)
        move_all_files_to_parent(
            self.dataset_path,
            "TAU-urban-acoustic-scenes-2020-mobile-development")
        self.set_dataset_download_finish()


def get_available_datasets():
    availabe_datasets = {m[0]: m[1] for m in inspect.getmembers(
        sys.modules[__name__], inspect.isclass) if m[1].__module__ == __name__}

    return availabe_datasets
