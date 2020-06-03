import pandas as pd
import os
#import sed_eval
import gzip
import glob
import pickle
import librosa
import soundfile as sf
import csv
import shutil
import sox

import numpy as np
import keras
import random
import math
from scipy.signal import hanning

from ..utils.ui import progressbar
from ..utils.data import get_fold_val
from ..utils.files import mkdir_if_not_exists, download_files_and_unzip, move_all_files_to_parent
import time     

    
class DataGenerator():
    """
    DataGenerator includes functions to load and manage DCASE datasets.
    This class can be redefined by child classes (see UrbanSound8k, ESC50)

    Attributes
    ----------
    audio_folder : str
        Path to the folder with audio files
    features_folder : str 
        Path to the folder with the features files    
    annotations_folder : str 
        Path to the folder with the annotation files    
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
        Form: {'fold1' : {'X' : [features1, features2, ...], 'Y': [ann1, ann2, ...]}, ...}

    Methods
    -------
    get_file_lists()
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
        Similar to get_data_for_training, but returns only one example (sequence)
        for each file
    return_file_list()
        Returns self.file_lists
    """
    def __init__(self, dataset_path, features_folder, features, audio_folder=None, use_validate_set=True):
        """ Initialize the DataGenerator 
        Parameters
        ----------
        audio_folder : str
            Path to the folder with audio files
        features_folder : str 
            Path to the folder with the features files    
        annotations_folder : str 
            Path to the folder with the annotation files    
        features : list of str
            Names of features to be loaded    
        fold_list : list of str 
            List of fold names
        label_list : list of str
            List of label names
        meta_file : str, optional
            Path to the metadata file of dataset, default None

        """
        # General attributes
        self.dataset_path = dataset_path
        self.features = features
        self.use_validate_set = use_validate_set 

        if audio_folder is None:
            self.audio_folder = os.path.join(dataset_path, 'audio')
        else:
            self.audio_folder = os.path.join(dataset_path, audio_folder)
        self.features_folder = os.path.join(dataset_path, features_folder)

        # make features folder if does not exists
        mkdir_if_not_exists(self.features_folder)
        
        # Specific attributes
        self.set_specific_attributes()

        self.file_lists = {}
        self.data = {}
        self.get_file_lists()

    def set_specific_attributes(self):
        self.fold_list = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"]
        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]
      #  self.meta_file = None
      #  self.taxonomy_file = None
        self.evaluation_mode = 'cross-validation'

        self.folders_list = []
        for fold in self.fold_list:
            self.folders_list.append({'audio': os.path.join(self.audio_folder, fold), 
                                      'features': os.path.join(self.features_folder, fold)})

    def get_file_lists(self):
        """ Create self.file_lists, a dict thath includes a list of files per fold 
        """
        for fold in self.fold_list:
            features_folder = os.path.join(self.features_folder, fold, self.features) 
            self.file_lists[fold] = sorted(glob.glob(os.path.join(features_folder, '*.npy')))

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
            y = self.get_annotations(file_name, features)
            annotations.append(y)

        return features_list, annotations

    def load_data(self):
        """ Creates self.data that contains features and annotations for all files
        in all folds
        """
        for fold in progressbar(self.fold_list, prefix='fold: '):
            X,Y = self.data_generation(self.file_lists[fold])
            self.data[fold] = {'X': X, 'Y': Y}

    def get_data_for_training(self, fold_test, upsampling=False):
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
        # cross-validation mode
        fold_val = get_fold_val(fold_test, self.fold_list)
        folds_train = self.fold_list.copy() #list(range(1,N_folds+1))
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

        X_train = np.concatenate(X_train_list,axis=0)
        Y_train = np.concatenate(Y_train_list,axis=0)

        X_train_up = X_train.copy()
        Y_train_up = Y_train.copy()

        ## upsampling
        if upsampling:
            n_classes = Y_train.shape[1]
            Ns = np.zeros(n_classes)
            for j in range(n_classes):
                Ns[j] = np.sum(Y_train[:,j]==1)
            Ns = np.floor(np.amax(Ns)/Ns)-1
            for j in range(n_classes):
                if Ns[j] > 1:
                    X_j = X_train[Y_train[:,j]==1]
                    Y_j = Y_train[Y_train[:,j]==1]
                    X_train_up = np.concatenate([X_train_up]+[X_j]*int(Ns[j]),axis=0)
                    Y_train_up = np.concatenate([Y_train_up]+[Y_j]*int(Ns[j]),axis=0)

        if self.use_validate_set:        
            return X_train_up, Y_train_up, X_val, Y_val
        else:
            return X_train_up, Y_train_up, X_train_list, Y_train_list

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
        X_test = self.data[fold_test]['X']
        Y_test = self.data[fold_test]['Y']

        return X_test, Y_test

    def get_one_example_per_file(self, fold_test):
        # cross-validation mode
        fold_val = get_fold_val(fold_test, self.fold_list)
        folds_train = self.fold_list.copy()
        folds_train.remove(fold_test)
        folds_train.remove(fold_val)

        X_val = self.data[fold_val]['X']
        Y_val = self.data[fold_val]['Y']

        X_train = []
        Y_train = []
        Files_names_train = []
        for fold_train in folds_train:
            for file in range(len(self.data[fold_train]['X'])):
                X = self.data[fold_train]['X'][file]
                if len(X) <= 1:
                    continue
                ix = int(len(X)/2)
                X = np.expand_dims(self.data[fold_train]['X'][file][ix],axis=0)
                X_train.append(X)
                Y = np.expand_dims(self.data[fold_train]['Y'][file][ix],axis=0)
                Y_train.append(Y)
                if self.file_lists is not None:
                    Files_names_train.append(self.file_lists[fold_train][file])
        X_train = np.concatenate(X_train,axis=0)
        Y_train = np.concatenate(Y_train,axis=0)     

        return X_train, Y_train, Files_names_train

    def return_file_list(self, fold_test):
        return self.file_lists[fold_test]

    def get_folder_lists(self):
        return self.folders_list

    def download_dataset(self, dataset_folder, zenodo_url, zenodo_files):
        download_files_and_unzip(dataset_folder, zenodo_url, zenodo_files)




# UrbanSound8k class (just a copy of DataGenerator)
import csv
class UrbanSound8k(DataGenerator):
    def __init__(self, dataset_path, features_folder, features, audio_folder=None, use_validate_set=True):
        super().__init__(dataset_path, features_folder, features, audio_folder, use_validate_set)

    def download_dataset(self, dataset_folder):
        zenodo_url = "https://zenodo.org/record/1203745/files"
        zenodo_files = ["UrbanSound8K.tar.gz"]
        super().download_dataset(dataset_folder, zenodo_url, zenodo_files)
        move_all_files_to_parent(dataset_folder, "UrbanSound8K") 

# ESC50 cllass
class ESC50(DataGenerator):
    def __init__(self, dataset_path, features_folder, features, audio_folder=None, use_validate_set=True):
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
                self.metadata[filename] = {'fold': fold, 'class_ix': class_ix, 'class_name': class_name, 'esc10': esc10}
                if class_name not in self.label_list:
                    self.label_list[class_ix] = class_name

        super().__init__(dataset_path, features_folder, features, audio_folder, use_validate_set)

    def set_specific_attributes(self):
        self.fold_list = ["fold1", "fold2", "fold3", "fold4", "fold5"]
        self.evaluation_mode = 'cross-validation'

        # all wav files are in the same folder
        self.folders_list = [{'audio': os.path.join(self.audio_folder),
                              'features': os.path.join(self.features_folder)}]

    def get_file_lists(self):
        self.file_lists = {}
        for fold in self.fold_list:
            self.file_lists[fold] = []
            features_folder = os.path.join(self.features_folder, self.features)   
            all_files = sorted(glob.glob(os.path.join(features_folder, '*.npy')))
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

    def download_dataset(self, dataset_folder):
        github_url = "https://github.com/karoldvl/ESC-50/archive/"
        github_files = ["master.zip"]
        super().download_dataset(dataset_folder, github_url, github_files)
        move_all_files_to_parent(dataset_folder, "ESC-50-master") 


class ESC10(ESC50):
    def __init__(self, dataset_path, features_folder, features, audio_folder=None, use_validate_set=True): 
        # first call init of ESC50 class
        super().__init__(dataset_path, features_folder, features, audio_folder, use_validate_set)
        
        # then change self.metadata and self.laberl_lsit to keep only ESC-10
        new_metada = {}
        new_label_list_ids = []
        for j in self.metadata.keys():
            if self.metadata[j]['esc10'] == True:
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
            assert self.metadata[j]['esc10'] == True   
            self.metadata[j]['class_ix'] = [i for i,x in enumerate(self.label_list) if x == self.metadata[j]['class_name']][0]

        # regenerate self.file_lists
        self.get_file_lists()

from pandas import read_csv
from sed_eval.util.event_roll import event_list_to_event_roll
# URBAN-SED class
class URBAN_SED(DataGenerator):
    def __init__(self, dataset_path, features_folder, features, audio_folder=None, use_validate_set=True, 
                 sequence_time=1.0,sequence_hop_time=0.5, metric_resolution_sec=1.0): 
        self.sequence_time = sequence_time
        self.sequence_hop_time = sequence_hop_time
        self.metric_resolution_sec = metric_resolution_sec
        super().__init__(dataset_path, features_folder, features, audio_folder, use_validate_set)

    def set_specific_attributes(self):
        self.annotations_folder = os.path.join(self.dataset_path, 'annotations')
        self.fold_list = ["train", "validate", "test"]
        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]
        self.evaluation_mode = 'train-validate-test'

        self.folders_list = []
        for fold in self.fold_list:
            self.folders_list.append({'audio': os.path.join(self.audio_folder, fold), 
                                      'features': os.path.join(self.features_folder, fold)})

    def get_file_lists(self):
        super().get_file_lists()
        self.features_to_labels = {}
        self.features_to_wav = {}
        for fold in self.fold_list:
            for fil in self.file_lists[fold]:
                label_file = os.path.basename(fil).split('.')[0] + '.txt'
                self.features_to_labels[fil] = os.path.join(self.annotations_folder, fold, label_file)
                audio_file = os.path.basename(fil).split('.')[0] + '.wav'
                self.features_to_wav[fil] = os.path.join(self.audio_folder, fold, audio_file)

    def data_generation(self, list_files_temp):
        features_list = []
        annotations_sequences = []
        annotations_grid_metrics = []
        for file_name in list_files_temp: 
            features = np.load(file_name)
            features_list.append(features)               
            y_sequences = self.get_annotations(file_name, features, time_resolution=self.sequence_hop_time)
            y_grid_metrics = self.get_annotations(file_name, features, time_resolution=self.metric_resolution_sec)
            #print(y_sequences.shape,y_grid_metrics.shape,features.shape)
            y_sequences = y_sequences[:len(features)]
            #print(y_sequences.shape, features.shape)
            assert y_sequences.shape[0] == features.shape[0]
            #print('seq', y_sequences.shape, 'grid', y_grid_metrics.shape)
            annotations_sequences.append(y_sequences)
            annotations_grid_metrics.append(y_grid_metrics)
            #annotations.append({'y_frames': y_frames, 'y_grid_metrics': y_grid_metrics})

        return features_list, [annotations_sequences, annotations_grid_metrics] 
    
    def get_annotations(self, file_name, features, time_resolution=1.0):
        label_file = self.features_to_labels[file_name]
        audio_file = self.features_to_wav[file_name]
        f = sf.SoundFile(audio_file)
        audio_len_sec = len(f) / f.samplerate
        labels = read_csv(label_file, delimiter='\t', header=None)
        labels.columns = ['event_onset', 'event_offset','event_label']

        #print(features.shape[0], self.sequence_hop_time, time_resolution)
        N_seqs = int(np.floor((audio_len_sec + self.sequence_time) / time_resolution))
        event_roll = np.zeros((N_seqs,len(self.label_list)))
        #print(event_roll.shape)
        for event in labels.to_dict('records'):
            pos = self.label_list.index(event['event_label'])
            
            event_onset = event['event_onset']
            event_offset = event['event_offset']

            #event_offset = 5.0
            #event_onset = 0.0

            onset = int(np.round(event_onset * 1 / float(time_resolution))) #math.floor
            offset = int(np.round(event_offset * 1 / float(time_resolution))) + 1 #math.ceil
            #print(event_onset,event_offset,onset,offset)
            event_roll[onset:offset, pos] = 1 
        return event_roll
 
    def get_data_for_training(self, fold_test='test'):
        # train-val-test mode
        X_val = self.data['validate']['X']
        Y_val = self.data['validate']['Y'][1] #grid time of metrics

        X_train = np.concatenate(self.data['train']['X'],axis=0)
        Y_train = np.concatenate(self.data['train']['Y'][0],axis=0) #grid time of instances for training

        return X_train, Y_train, X_val, Y_val

    def get_data_for_testing(self, fold_test='test'):
        # train-val-test mode
        X_test = self.data[fold_test]['X']
        Y_test = self.data[fold_test]['Y'][1]

        return X_test, Y_test

    def download_dataset(self, dataset_folder):
        zenodo_url = "https://zenodo.org/record/1324404/files"
        zenodo_files = ["URBAN-SED_v2.0.0.tar.gz"]

        super().download_dataset(dataset_folder, zenodo_url, zenodo_files)
        move_all_files_to_parent(dataset_folder, "URBAN-SED_v2.0.0")     

from pandas import read_csv
import yaml
class SONYC_UST(DataGenerator):
    def __init__(self, dataset_path, features_folder, features, audio_folder=None, use_validate_set=True): 
        super().__init__(dataset_path, features_folder, features, audio_folder, use_validate_set)

    def set_specific_attributes(self):
        self.fold_list = ["train", "validate"]
        self.evaluation_mode = 'train-validate-test'
        self.meta_file = os.path.join(self.dataset_path, 'annotations.csv')
        self.taxonomy_file = os.path.join(self.dataset_path, 'dcase-ust-taxonomy.yaml')

        self.metadata = read_csv(self.meta_file).sort_values('audio_filename')

        with open(self.taxonomy_file, 'r') as f:
            self.label_list = yaml.load(f, Loader=yaml.Loader)

        # all wav files are in the same folder
        self.folders_list = [{'audio': os.path.join(self.audio_folder),
                              'features': os.path.join(self.features_folder)}]

    def get_file_lists(self):
        filename_to_split = self.metadata[['audio_filename','split']].drop_duplicates()
        all_files_in_metadata = filename_to_split['audio_filename'].to_list()
        splits = filename_to_split['split'].to_list()

        self.file_lists = {}
        for fold in self.fold_list:
            self.file_lists[fold] = []
            features_folder = os.path.join(self.features_folder, self.features)   
            all_files = sorted(glob.glob(os.path.join(self.audio_folder, '*.wav')))
            assert len(all_files) != 0
            for fil in all_files:
                basename = os.path.basename(fil)
                if basename in all_files_in_metadata:
                    j = all_files_in_metadata.index(basename)
                    if splits[j] == fold:
                        file_npy = os.path.join(features_folder, basename.split('.')[0] + '.npy')
                        self.file_lists[fold].append(file_npy) 

    def get_annotations(self, file_name, features):
       # only coarse level
       # TODO add fine level
        n_classes_coarse_level = len(self.label_list['coarse'])
        y = np.zeros(n_classes_coarse_level)
        basename = os.path.basename(file_name).split('.')[0] + '.wav'

        metadata_of_file = self.metadata[self.metadata['audio_filename'] == basename]
        for class_ix in self.label_list['coarse']:
            class_column = str(class_ix) + '_' + self.label_list['coarse'][class_ix] + '_presence'
            # class present if any annotator check presence
            y[class_ix-1] = np.sum(metadata_of_file[class_column].values) >= 1

        y = np.expand_dims(y, 0)
        y = np.repeat(y, len(features), 0)
        return y

    def get_data_for_training(self, fold_test='test'):
        # train-val-test mode
        X_val = self.data['validate']['X']
        Y_val = self.data['validate']['Y']

        X_train = np.concatenate(self.data['train']['X'],axis=0)
        Y_train = np.concatenate(self.data['train']['Y'],axis=0)

        return X_train, Y_train, X_val, Y_val

    def get_data_for_testing(self, fold_test='test'):
        return None, None

    def download_dataset(self, dataset_folder):
        zenodo_url = "https://zenodo.org/record/3693077/files"
        zenodo_files = ["annotations.csv", "audio.tar.gz", "dcase-ust-taxonomy.yaml", "README.md"]
        super().download_dataset(dataset_folder, zenodo_url, zenodo_files)



class TAUUrbanAcousticScenes2020Mobile(DataGenerator):
    def __init__(self, dataset_path, features_folder, features, audio_folder=None, use_validate_set=True): 
        super().__init__(dataset_path, features_folder, features, audio_folder, use_validate_set)

    def set_specific_attributes(self):
        self.fold_list = ["train", "test"]
        self.meta_file = os.path.join(self.dataset_path, 'meta.csv')
        self.label_list = ['airport', 'shopping_mall', 'metro_station', 
                           'street_pedestrian', 'public_square', 'street_traffic',
                           'tram', 'bus', 'metro', 'park']

        self.evaluation_setup_train = os.path.join(self.dataset_path, 'evaluation_setup', 'fold1_train.csv')
        self.evaluation_setup_test = os.path.join(self.dataset_path, 'evaluation_setup', 'fold1_test.csv')
        self.annotations_folder = os.path.join(self.dataset_path, 'annotations')

        # all wav files are in the same folder
        self.folders_list = [{'audio': os.path.join(self.audio_folder),
                              'features': os.path.join(self.features_folder)}]

    def get_file_lists(self):
        self.file_lists = {}
        evaluation_files = [self.evaluation_setup_train, self.evaluation_setup_test]
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
                    file_name_npy = os.path.basename(file_name).split('.')[0] + '.npy'
                    self.file_lists[fold].append(os.path.join(self.features_folder, self.features, file_name_npy))


    def get_annotations(self, file_name, features):
        y = np.zeros((len(features), len(self.label_list)))
        basename = os.path.basename(file_name)
        # delete file extension
        basename = basename.split('.')[0]
        scene_label, city, location_id, segment_id, device_id = basename.split('-')
        class_ix = self.label_list.index(scene_label)
        y[:, class_ix] = 1
        return y

    def get_data_for_training(self, fold_test='test'):
        # train-val-test mode

        X_train = np.concatenate(self.data['train']['X'],axis=0)
        Y_train = np.concatenate(self.data['train']['Y'],axis=0)

        # TODO: make a validation set
        X_val = X_train
        Y_val = Y_train

        return X_train, Y_train, X_val, Y_val

    def get_data_for_testing(self, fold_test='test'):
        X_test = np.concatenate(self.data['test']['X'],axis=0)
        Y_test = np.concatenate(self.data['test']['Y'],axis=0)        
        return X_test, Y_test


    def download_dataset(self, dataset_folder):
        zenodo_url = "https://zenodo.org/record/3819968/files"
        zenodo_files = ["TAU-urban-acoustic-scenes-2020-mobile-development.audio.%d.zip" % j for j in range(1,17)]
        zenodo_files.append('TAU-urban-acoustic-scenes-2020-mobile-development.doc.zip')
        zenodo_files.append('TAU-urban-acoustic-scenes-2020-mobile-development.meta.zip')

        super().download_dataset(dataset_folder, zenodo_url, zenodo_files)
        move_all_files_to_parent(dataset_folder, "TAU-urban-acoustic-scenes-2020-mobile-development")  