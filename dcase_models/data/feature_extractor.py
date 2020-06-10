import pandas as pd
import os
import numpy as np
import librosa
import glob
import soundfile as sf
import json
import openl3

from ..utils.ui import progressbar
from ..utils.files import mkdir_if_not_exists, load_json


class FeatureExtractor():
    """
    FeatureExtractor includes functions to calculates features.
    This class can be inherited to customize (i.e. see MelSpectrogram, Openl3)

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5, audio_win=1024, audio_hop=512, n_fft=1024, sr=44100):
        """ Initialize the FeatureExtractor 
        Parameters
        ----------

        """
        self.sequence_time = sequence_time
        self.sequence_hop_time = sequence_hop_time
        self.audio_hop = audio_hop
        self.audio_win = audio_win
        self.n_fft = n_fft
        self.sr = sr

        self.sequence_frames = int(
            (sequence_time * sr - self.audio_win) / float(audio_hop))
        self.sequence_hop = int(sequence_hop_time * sr / float(audio_hop))

        self.params = {'sr': self.sr,
                       'sequence_time': self.sequence_time,
                       'sequence_hop_time': self.sequence_hop_time,
                       'audio_hop': self.audio_hop, 'audio_win': self.audio_win,
                       'n_fft': self.n_fft, 'features': 'FeatureExtractor'}

    def get_sequences(self, x, pad=True):
        """ Extract sequences (windows) of a 2D representation
        Parameters
        ----------
        x : ndarray
            2D representation
        pad : bool 
            if True, pad x before windowing

        Returns
        -------
        list of ndarray
            list of sequences      

        """
        if pad:
            x = np.pad(x, ((0, 0), (self.sequence_frames//2,
                                    self.sequence_frames//2)), 'reflect')
        hop_times = np.arange(
            0, x.shape[1]-self.sequence_frames+1, self.sequence_hop)

        y = []
        for i in hop_times:
            x_seq = x[:, i:i+self.sequence_frames]
            y.append(x_seq)

        return y

    def load_audio(self, file_name, mono=True, change_sampling_rate=True):
        """ Load an audio signal and convert to mono if needed

        Parameters
        ----------
        file_name : str
            Path to the audio file
        mono : bool 
            if True, only returns left channel

        Returns
        -------
        array
            audio signal

        """
        audio, sr_old = sf.read(file_name)

        # convert to mono
        if (len(audio.shape) > 1) & (mono):
            audio = audio[:, 0]

        # continuous array (for some librosa functions)
        audio = np.asfortranarray(audio)

        if (self.sr != sr_old) & (change_sampling_rate):
            print('changing sampling rate', sr_old, self.sr)
            audio = librosa.resample(audio, sr_old, self.sr)

        return audio

    def calculate_features(self, file_name):
        """ Calculates features of an audio file

        Parameters
        ----------
        file_name : str
            Path to the audio file

        Returns
        -------
        ndarray
            feature representation of the audio signal

        """

        audio = self.load_audio(file_name)

        # spectrogram
        stft = librosa.core.stft(audio, n_fft=self.n_fft, hop_length=self.audio_hop,
                                 win_length=self.audio_win, center=True)

        # power
        spectrogram = np.abs(stft)**2

        # convert to sequences (windowing)
        spectrogram_seqs = self.get_sequences(spectrogram, pad=True)

        # convert to numpy
        spectrogram_np = np.asarray(spectrogram_seqs)

        # transpose time and freq dims
        spectrogram_np = np.transpose(spectrogram_np, (0, 2, 1))

        return spectrogram_np

    def extract(self, folder_audio, folder_features):
        """ Extract feauters for all files present in folder_audio

        Parameters
        ----------
        folder_audio : str
            Path to the audio folder
        folder_features : str 
            Path to the feature folder.

        """
        feature_name = self.params['name']
        feature_path = os.path.join(folder_features, feature_name)

        # if features already were extracted, exit.
        if self.check_features_folder(feature_path):
            return None

        mkdir_if_not_exists(folder_features)
        mkdir_if_not_exists(feature_path)

        files_orig = sorted(glob.glob(os.path.join(folder_audio, '*.wav')))

        for file_audio in progressbar(files_orig, "Computing: ", 40):

            features_array = self.calculate_features(file_audio)
            # print(spectrograms.shape)
            file_features = file_audio.split('/')[-1]
            file_features = file_features.replace('wav', 'npy')
            np.save(os.path.join(feature_path, file_features), features_array)

        self.save_parameters_json(feature_path)
        return 1

    def save_parameters_json(self, path):
        """ Save a json file with the self.params. Useful for checking if 
        the features files were calculated with same parameters.

        Parameters
        ----------
        path : str
            Path to the JSON file

        """
        json_path = os.path.join(path, "parameters.json")
        with open(json_path, 'w') as fp:
            json.dump(self.params, fp)

    def check_features_folder(self, features_folder):
        json_features_folder = os.path.join(features_folder, "parameters.json")
        print(json_features_folder)
        if not os.path.exists(json_features_folder):
            return False
        parameters_features_folder = load_json(json_features_folder)

        for key in parameters_features_folder.keys():
            if parameters_features_folder[key] != self.params[key]:
                return False

        return True


class Spectrogram(FeatureExtractor):
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5, audio_win=1024, audio_hop=512, n_fft=1024, sr=44100):

        super().__init__(sequence_time=sequence_time, sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop, n_fft=n_fft, sr=sr)

        self.params['name'] = 'Spectrogram'


class MelSpectrogram(FeatureExtractor):
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5, audio_win=1024, audio_hop=512,
                 n_fft=1024, sr=44100, mel_bands=128, fmax=None):

        super().__init__(sequence_time=sequence_time, sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop, n_fft=n_fft, sr=sr)

        self.params['name'] = 'MelSpectrogram'
        self.params['mel_bands'] = mel_bands
        self.params['fmax'] = fmax

        self.mel_basis = librosa.filters.mel(
            sr, n_fft, mel_bands, htk=True, fmax=fmax)

    def calculate_features(self, file_name):
        # get spectrograms
        #spectrograms = super().calculate_features(file_name)

        # load audio
        audio = self.load_audio(file_name)
        # spectrogram
        stft = librosa.core.stft(audio, n_fft=self.n_fft, hop_length=self.audio_hop,
                                 win_length=self.audio_win, center=True)
        # power
        spectrogram = np.abs(stft)**2
        # convert to mel_spectrogram

        mel_spectrogram = self.mel_basis.dot(spectrogram)
        assert mel_spectrogram.shape[0] == self.params['mel_bands']

        # convert power to db
        mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # convert to sequences (windowing)
        mel_spectrogram_seqs = self.get_sequences(mel_spectrogram, pad=True)

        # convert to numpy
        mel_spectrogram_np = np.asarray(mel_spectrogram_seqs)

        # transpose time and freq dims
        mel_spectrogram_np = np.transpose(mel_spectrogram_np, (0, 2, 1))

        return mel_spectrogram_np


class Openl3(FeatureExtractor):
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5, audio_win=1024, audio_hop=512,
                 n_fft=1024, sr=44100, content_type="env", input_repr="mel256", embedding_size=512):

        super().__init__(sequence_time=sequence_time, sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop, n_fft=n_fft, sr=sr)

        self.params['name'] = 'Openl3'
        self.params['content_type'] = content_type
        self.params['input_repr'] = input_repr
        self.params['embedding_size'] = embedding_size

    def calculate_features(self, file_name):
        audio = self.load_audio(file_name, change_sampling_rate=False)
        emb, ts = openl3.get_audio_embedding(audio, self.sr,
                                             content_type=self.params['content_type'],
                                             embedding_size=self.params['embedding_size'],
                                             input_repr=self.params['input_repr'],
                                             hop_size=self.sequence_hop_time,
                                             verbose=False)

        return emb
