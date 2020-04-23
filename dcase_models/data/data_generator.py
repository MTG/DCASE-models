import pandas as pd
import os
#import sed_eval
import gzip
import glob
import pickle
import librosa
import soundfile as sf
import csv

import numpy as np
import keras
import random
import math
from scipy.signal import hanning

from ..utils.ui import progressbar
from ..utils.data import get_fold_val
import time     

    
class DataGenerator():
    'Generates data for the experiments'
    def __init__(self, audio_folder, features_folder, annotations_folder, features, fold_list, label_list, meta_file=None):

        """ Initialize the DataGenerator 
        Parameters
        ----------
        list_IDs : list

        """
        self.audio_folder = audio_folder
        self.features_folder = features_folder
        self.annotations_folder = annotations_folder
        self.features = features
        self.fold_list = fold_list
        self.label_list = label_list
        self.get_file_lists()

    def get_file_lists(self):
        self.file_lists = {}
        for fold in self.fold_list:
            features_folder = os.path.join(self.features_folder, fold, self.features)   
            self.file_lists[fold] = sorted(glob.glob(os.path.join(features_folder, '*.npy')))

    def get_annotations(self,file_name, features):
        y = np.zeros((len(self.label_list)))
        class_ix = int(os.path.basename(file_name).split('-')[1])
        y[class_ix] = 1
        y = np.expand_dims(y, 0)
        y = np.repeat(y, len(features), 0)
        return y

    def data_generation(self, list_files_temp):
        """ This function generates data with the files in list_IDs_temp
        
        Parameters
        ----------
        list_IDs_temp : list
            List of file IDs.

        Return
        ----------
        X : array
            Audio signals (only for end-to-end networks)
            
        S : array
            Spectrograms
            
        mel : array
            Mel-spectrograms
           
        yt : array
            Annotationes as categorical matrix

        """
        
        features_list = []
        annotations = []

        for file_name in list_files_temp: 
            features = np.load(file_name)
            #(features.shape)
            features_list.append(features)               
            y = self.get_annotations(file_name, features)
            annotations.append(y)

        return features_list, annotations

    def load_data(self):
        self.data = {}
        for fold in progressbar(self.fold_list, prefix='fold: '):
            #print(self.file_lists[fold])
            X,Y = self.data_generation(self.file_lists[fold])
            self.data[fold] = {'X': X, 'Y': Y}

    def get_data_for_training(self, fold_test):
        # cross-validation mode
        fold_val = get_fold_val(fold_test, self.fold_list)
        folds_train = self.fold_list.copy() #list(range(1,N_folds+1))
        folds_train.remove(fold_test)
        folds_train.remove(fold_val)

        X_val = self.data[fold_val]['X']
        Y_val = self.data[fold_val]['Y']

        X_train = []
        Y_train = []
        for fold in folds_train:
            X_train.extend(self.data[fold]['X'])
            Y_train.extend(self.data[fold]['Y'])

        X_train = np.concatenate(X_train,axis=0)
        Y_train = np.concatenate(Y_train,axis=0)

        return X_train, Y_train, X_val, Y_val

    def get_data_for_testing(self, fold_test):
        # cross-validation mode
        X_test = self.data[fold_test]['X']
        Y_test = self.data[fold_test]['Y']

        return X_test, Y_test

# UrbanSound8k class (just a copy of DataGenerator)
import csv
class UrbanSound8k(DataGenerator):
    def __init__(self, audio_folder, features_folder, annotations_folder, features, fold_list, label_list, meta_file=None):
        super().__init__(audio_folder, features_folder, annotations_folder, features, fold_list, label_list, meta_file)


# ESC50 cllass
class ESC50(DataGenerator):
    def __init__(self, audio_folder, features_folder, annotations_folder, features, fold_list, label_list, meta_file=None):
        self.metadata = {}
        n_classes = 50
        label_list = ['']*n_classes
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
                if class_name not in label_list:
                    label_list[class_ix] = class_name

        super().__init__(audio_folder, features_folder, annotations_folder, features, fold_list, label_list, meta_file)

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


class ESC10(ESC50):
    def __init__(self, audio_folder, features_folder, annotations_folder, features, fold_list, label_list, meta_file=None):
        # first call init of ESC50 class
        super().__init__(audio_folder, features_folder, annotations_folder, features, fold_list, label_list, meta_file)
        
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


# URBAN-SED class
class URBAN_SED(DataGenerator):
    def __init__(self, audio_folder, features_folder, annotations_folder, features, fold_list, label_list, sequence_hop_time, meta_file=None):
        self.sequence_hop_time = sequence_hop_time
        super().__init__(audio_folder, features_folder, annotations_folder, features, fold_list, label_list, meta_file)

    def get_file_lists(self):
        super().__init__()
        self.wav_to_labels = {}
        for fold in self.fold_list:
            for fil in self.file_lists[fold]:
                label_file = os.path.basename(filename).split('.')[0] + '.txt'
                self.wav_to_labels[fil] = os.path.join(self.annotations_folder, fold, label_file)

    from pandas import read_csv
    def get_annotations(self, file_name, features):
        label_file = self.wav_to_labels[file_name]
        labels = read_csv(label_file, delimiter='\t', header=None)
        labels.columns = ['event_onset', 'event_offset','event_label']
        event_roll = np.zeros((features.shape[0],len(self.label_list)))
        for event in labels.to_dict('records'):
            pos = self.label_list.index(event['event_label'])
            
            event_onset = event['event_onset']
            event_offset = event['event_offset']
            
            onset = int(math.floor(event_onset * 1 / float(self.sequence_hop_time)))
            offset = int(math.ceil(event_offset * 1 / float(self.sequence_hop_time)))
            
            event_roll[onset:offset, pos] = 1 
        return event_roll
 