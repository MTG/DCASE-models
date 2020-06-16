import os
import numpy as np
import librosa
import glob
import soundfile as sf
import json

from ..utils.ui import progressbar
from ..utils.files import mkdir_if_not_exists, load_json


class FeatureExtractor():
    """
    Abstract base class for feature extraction. 
    
    Includes methods to load audio files, calculates features and
    prepare sequences. 

    Inherit this class to define custom features 
    (e.g. features.MelSpectrogram, features.Openl3).

    Attributes
    ----------
    sequence_time : float
        Duration of the sequence analysis window (in seconds).
    sequence_hop_time : float
        Hop time of the sequence analysis windows (in seconds).
    audio_win : int
        Number of samples of the audio frames.
    audio_hop : int
        Number of samples for the frame hop between adjacent 
        audio frames. 
    sr : int
        Sampling rate of the audio signals
        If the original audio is not sampled at this rate,
        it is re-sampled before feature extraction   
    sequence_frames : int
        Number of frames equivalent to the sequence_time.
    sequence_hop : int
        Number of frames equivalent to the sequence_hop_time.
    params : dict
        Dictionary that stores important parameters.
        This is used to check if the features were extracted before.

    Methods
    -------
    get_sequences(x, pad=True)
        Extract sequences (windows) of a the feature representation.
    load_audio(file_name, mono=True, change_sampling_rate=True)
        Load an audio signal and convert to mono if needed.
    calculate_features(file_name)
        Load an audio file and calculate features.
    save_parameters_json(path)
        Save a json file with the params.
    check_features_folder(features_folder)
        Checks if the features were calculated before.
    get_features_shape(length_sec=10.0)
        Run calculate_features with a dummy signal of length length_sec
        and returns the shape of the feature representation.

    """

    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=512, sr=44100):
        """ Initialize the FeatureExtractor
        Parameters
        ----------
        sequence_time : float
            Duration of the sequence analysis window (in seconds)
        sequence_hop_time : float
            Hop time of the sequence analysis windows (in seconds)
        audio_win : int
            Number of samples of the audio frames.
        audio_hop : int
            Number of samples for the frame hop between adjacent 
            audio frames   
        sr : int
            Sampling rate of the audio signals
            If the original audio is not sampled at this rate,
            it is re-sampled before feature extraction       
        
        """
        self.sequence_time = sequence_time
        self.sequence_hop_time = sequence_hop_time
        self.audio_hop = audio_hop
        self.audio_win = audio_win
        self.sr = sr

        self.sequence_frames = int(
            (sequence_time * sr - self.audio_win) / float(audio_hop))
        self.sequence_hop = int(sequence_hop_time * sr / float(audio_hop))

        self.params = {'sr': self.sr,
                       'sequence_time': self.sequence_time,
                       'sequence_hop_time': self.sequence_hop_time,
                       'audio_hop': self.audio_hop,
                       'audio_win': self.audio_win,
                       'features': 'FeatureExtractor'}

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
        change_sampling_rate : bool
            if True, the audio signal is re-sampled to self.sr

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
            print('Changing sampling rate from %d to %d' % (sr_old, self.sr))
            audio = librosa.resample(audio, sr_old, self.sr)

        return audio

    def calculate_features(self, file_name):
        """ 
        Load an audio file and calculates features

        Parameters
        ----------
        file_name : str
            Path to the audio file

        Returns
        -------
        ndarray
            feature representation of the audio signal

        """
        pass


    def save_parameters_json(self, path):
        """ 
        Save a json file with the self.params. 
        
        Useful for checking if the features files were calculated
        with same parameters.

        Parameters
        ----------
        path : str
            Path to the JSON file

        """
        json_path = os.path.join(path, "parameters.json")
        with open(json_path, 'w') as fp:
            json.dump(self.params, fp)

    def check_features_folder(self, features_folder):
        """ Checks if the features saved in features_folder were
        calculated with the same parameters of self.params

        Parameters
        ----------
        path : str
            Path to the features folder

        """

        json_features_folder = os.path.join(features_folder, "parameters.json")
        if not os.path.exists(json_features_folder):
            return False
        parameters_features_folder = load_json(json_features_folder)

        for key in parameters_features_folder.keys():
            if parameters_features_folder[key] != self.params[key]:
                return False

        return True

    def get_features_shape(self, length_sec=10.0):
        """
        Run calculate_features() with a dummy signal of length length_sec
        and returns the shape of the feature representation.

        Parameters
        ----------
        length_sec : float
            Duration in seconds of the test signal

        Returns
        ----------       
        tuple
            Shape of the feature representation
        """

        audio_sample = np.zeros(int(length_sec*self.sr))
        audio_file = 'zeros.wav'
        sf.write('zeros.wav', audio_sample, self.sr)
        features_sample = self.calculate_features(audio_file)
        os.remove(audio_file)
        return features_sample.shape