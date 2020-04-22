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
    def __init__(self, audio_folder, features_folder, features, fold_list, label_list):

        """ Initialize the DataGenerator 
        Parameters
        ----------
        list_IDs : list

        """
        self.audio_folder = audio_folder
        self.features_folder = features_folder
        self.features = features
        self.fold_list = fold_list
        self.label_list = label_list
        self.get_file_lists()

    def get_file_lists(self):
        self.file_lists = {}
        for fold in self.fold_list:
            features_folder = os.path.join(self.features_folder, fold, self.features)   
            self.file_lists[fold] = sorted(glob.glob(os.path.join(features_folder, '*.npy')))

    def __get_annotations(self,file_name, features):
        y = np.zeros((len(self.label_list)))
        class_ix = int(os.path.basename(file_name).split('-')[1])
        y[class_ix] = 1
        y = np.expand_dims(y, 0)
        y = np.repeat(y, len(features), 0)
        return y

    def __data_generation(self, list_files_temp):
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
            y = self.__get_annotations(file_name, features)
            annotations.append(y)

        return features_list, annotations

    def load_data(self):
        self.data = {}
        for fold in progressbar(self.fold_list, prefix='fold: '):
            #print(self.file_lists[fold])
            X,Y = self.__data_generation(self.file_lists[fold])
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
class UrbanSound8k(DataGenerator):
    def __init__(self, audio_folder, features_folder, features, fold_list, label_list):
        super().__init__(audio_folder, features_folder, features, fold_list, label_list)