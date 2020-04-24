import pandas as pd
import os
import numpy as np
import librosa
import glob
import soundfile as sf
import json
from ..utils.ui import progressbar
from ..utils.files import mkdir_if_not_exists

class FeatureExtractor():
    
    def __init__(self,sequence_time=1.0, sequence_hop_time=0.5,
                 audio_hop=882, audio_win=1764, n_fft=2048, sr=44100, mel_bands=128,
                 features=['spec','melspec'], fmax=None, augmentation=None):

        self.sr = sr
        self.n_fft = n_fft
        self.mel_bands = mel_bands
        self.audio_hop = audio_hop
        self.audio_win = audio_win
        self.mel_basis = librosa.filters.mel(sr,n_fft,mel_bands,htk=True,fmax=fmax)
        self.sequence_frames = int( (sequence_time * sr - self.audio_win) / float(audio_hop))
        self.sequence_hop = int(sequence_hop_time * sr / float(audio_hop))
        self.sequence_samples = int(sequence_time * sr)
        self.hop_time = audio_hop / float(sr)
        self.sequence_hop_time = sequence_hop_time
        self.sequence_time = sequence_time
        self.features = features
        self.augmentation = augmentation
        if self.augmentation is not None:
            self.augmentation_type = list(self.augmentation.keys())[0]
            self.augmentation_param = self.augmentation[self.augmentation_type]
        print(self.sequence_frames, self.sequence_hop)

    def calculate_features(self, file_name):
            audio,sr_old = sf.read(file_name)
            if len(audio.shape) > 1:
                audio = audio[:,0]
                
            if self.sr != sr_old:
                print('changing sampling rate',sr_old,self.sr)
                audio = librosa.resample(audio, sr_old, self.sr)                    
              
            audio = np.asfortranarray(audio)

            if self.augmentation is not None:
                if self.augmentation_type == 'pitch_shift':
                    audio = librosa.effects.pitch_shift(audio, self.sr, self.augmentation_param)

            # complex stft
            stft = librosa.core.stft(audio, n_fft=self.n_fft, hop_length=self.audio_hop,
                                                win_length=self.audio_win, center=False)
            
           # print(stft.shape)
            if self.augmentation is not None:
                if self.augmentation_type == 'time_stretch':
                    stft = librosa.core.phase_vocoder(stft, rate=self.augmentation_param)

            # stft padding
            pad = 0
            if stft.shape[1] < self.sequence_frames:
                #stft = np.pad(stft, ((0,0),(0, self.sequence_frames-stft.shape[1])), 'constant', constant_values=(0, 0))
                #stft = np.pad(stft, ((0,0),(0, self.sequence_frames-stft.shape[1])), 'wrap')
               # print('stft',stft.shape,self.sequence_frames)
               # stft = librosa.core.phase_vocoder(stft, rate=stft.shape[1]/self.sequence_frames)
               pad = self.sequence_frames-stft.shape[1]
               stft = np.pad(stft, ((0,0),(0, self.sequence_frames-stft.shape[1])), 'constant', constant_values=(0,0))
               
                #print(stft.shape)

            # window padding
            expected_n_sequences = (stft.shape[1])/ float(self.sequence_hop)
            hop_times = np.arange(0,stft.shape[1]-self.sequence_frames+1,self.sequence_hop)
            if (expected_n_sequences > len(hop_times)) & (stft.shape[1] > self.sequence_frames):
                hop_times = np.concatenate((hop_times, (stft.shape[1]-self.sequence_frames,)),axis=0)

            # power
            stft = np.abs(stft)**2

            features_list = {}
            #spectrograms = []  

            for i in hop_times:
                spectrogram = stft[:,i:i+self.sequence_frames]

                for feature in self.features:
                    if feature not in features_list:
                        features_list[feature] = []

                    if feature == 'spectrogram':
                        features_list[feature].append(spectrogram.T)
                    if feature == 'mel_spectrogram':
                        melspec = self.mel_basis.dot(spectrogram)
                        melspec = librosa.core.power_to_db(melspec)
                        features_list[feature].append(melspec.T)

                    if feature == 'chroma':    
                        chroma = librosa.feature.chroma_stft(sr=self.sr,S=spectrogram)
                        features_list[feature].append(chroma.T)

                    if feature == 'tonnetz':
                        if chroma is None:
                            chroma = librosa.feature.chroma_stft(sr=self.sr,S=spectrogram)
                        tonnetz = librosa.feature.tonnetz(sr=self.sr, chroma=chroma)
                        features_list[feature].append(tonnetz.T)

                    if feature == 'spectral_contrast':     
                        spectral_contrast = librosa.feature.spectral_contrast(sr=self.sr,S=spectrogram)
                        features_list[feature].append(spectral_contrast.T)

                    # if feature is a function
                    if callable(feature):
                        sequence_samples = int(self.sequence_time * self.sr)
                        audio_slice = audio[i*self.audio_hop:i*self.audio_hop + sequence_samples]
                        custom_feature = feature(sr=self.sr,S=spectrogram, audio=audio_slice)
                        features_list[feature].append(custom_feature.T)

            for feature in self.features:
                features_list[feature] = np.asarray(features_list[feature])

            return features_list

    def extract(self, folder_audio, folder_features):
        mkdir_if_not_exists(folder_features)
        files_orig = sorted(glob.glob(os.path.join(folder_audio, '*.wav')))

        for file_audio in progressbar(files_orig, "Computing: ", 40):
            
            features_list = self.calculate_features(file_audio)
            #print(spectrograms.shape)
            file_features = file_audio.split('/')[-1]
            file_features = file_features.replace('wav','npy')
            
            for feature in self.features:
                feature_name = feature
                if callable(feature):
                    feature_name = feature.__name__

                if self.augmentation is not None:
                    folder_name = '%s_%s_%02.1f' % (feature_name, self.augmentation_type, self.augmentation_param)
                else:
                    folder_name = feature_name

                feature_path = os.path.join(folder_features,folder_name)
                mkdir_if_not_exists(feature_path)

                np.save(os.path.join(feature_path,file_features),features_list[feature])
                
    def save_mel_basis(self, path):
        np.save(path,self.mel_basis)

    def save_parameters_json(self, path):
        features = []
        for feature in self.features:
            if callable(feature):
                features.append(feature.__name__)
            else:
                features.append(feature)
                
        params = {'sr': self.sr, 'mel_bands': self.mel_bands,
                'sequence_time': self.sequence_time, 
                'sequence_hop_time': self.sequence_hop_time,
                'audio_hop': self.audio_hop, 'audio_win': self.audio_win,
                'n_fft': self.n_fft, 'features': features}

        with open(path, 'w') as fp:
            json.dump(params, fp)
