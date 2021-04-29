import os
import numpy as np
import librosa
import soundfile as sf
import json
from scipy.stats import kurtosis, skew

from dcase_models.util.files import load_json, mkdir_if_not_exists
from dcase_models.util.files import duplicate_folder_structure
from dcase_models.util.files import list_wav_files
from dcase_models.util.ui import progressbar


class FeatureExtractor():
    """ Abstract base class for feature extraction.

    Includes methods to load audio files, calculate features and
    prepare sequences.

    Inherit this class to define custom features
    (e.g. features.MelSpectrogram, features.Openl3).

    Parameters
    ----------
    sequence_time : float, default=1.0
        Length (in seconds) of the feature representation analysis
        windows (model's input).

    sequence_hop_time : float, default=0.5
        Hop time (in seconds) of the feature representation analysis windows.

    audio_win : int, default=1024
        Window length (in samples) for the short-time audio processing
        (e.g short-time Fourier Transform (STFT))

    audio_hop : int, default=680
        Hop length (in samples) for the short-time audio processing
        (e.g short-time Fourier Transform (STFT))

    sr : int, default=22050
        Sampling rate of the audio signals.
        If the original audio is not sampled at this rate, it is re-sampled
        before feature extraction.

    Attributes
    ----------
    sequence_frames : int
        Number of frames equivalent to the sequence_time.

    sequence_hop : int
        Number of frames equivalent to the sequence_hop_time.

    Examples
    --------
    To create a new feature representation, it is necessary to define a class
    that inherits from FeatureExtractor. It is required to define the
    calculate() method.::

        from dcase_models.data.feature_extractor import FeatureExtractor
        class Chroma(FeatureExtractor):
            def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                             audio_win=1024, audio_hop=512, sr=44100,
                             # Add here your custom parameters
                             n_fft=1024, n_chroma=12):
                # Don't forget this line
                super().__init__(sequence_time=sequence_time,
                                 sequence_hop_time=sequence_hop_time,
                                 audio_win=audio_win,
                                 audio_hop=audio_hop, sr=sr)

                self.sequence_samples = int(librosa.core.frames_to_samples(
                    self.sequence_frames,
                    self.audio_hop,
                    n_fft=self.n_fft
                ))
            def calculate(self, file_name):
                # Here define your function to calculate the chroma features
                # Load the audio signal
                audio = self.load_audio(file_name)
                # Pad audio signal
                audio = librosa.util.fix_length(
                    audio,
                    audio.shape[0] + self.sequence_samples,
                    axis=0, mode='constant'
                )
                # Get the chroma features
                chroma = librosa.feature.chroma_stft(y=audio,
                                                     sr=self.sr,
                                                     n_fft=self.n_fft,
                                                     hop_length=audio_hop,
                                                     win_length=audio_win
                                                     )
                # Convert to sequences
                chroma = np.ascontiguousarray(chroma)
                chroma = librosa.util.frame(chroma,
                                            self.sequence_frames,
                                            self.sequence_hop,
                                            axis=0
                                            )
                return chroma

    """

    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=680, sr=22050, **kwargs):
        """ Initialize the FeatureExtractor

        """
        self.sequence_time = sequence_time
        self.sequence_hop_time = sequence_hop_time
        self.audio_hop = audio_hop
        self.audio_win = audio_win
        self.sr = sr

        self.sequence_frames = int(librosa.core.time_to_frames(
            sequence_time, sr=sr, hop_length=audio_hop))
        self.sequence_hop = int(librosa.core.time_to_frames(
            sequence_hop_time, sr=sr, hop_length=audio_hop))

        self.features_folder = kwargs.get('features_folder', 'features')
        self.pad_mode = kwargs.get('pad_mode', None)

    def load_audio(self, file_name, mono=True, change_sampling_rate=True):
        """ Loads an audio signal and converts it to mono if needed

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

    def calculate(self, file_name):
        """ Loads an audio file and calculates features

        Parameters
        ----------
        file_name : str
            Path to the audio file

        Returns
        -------
        ndarray
            feature representation of the audio signal

        """
        raise NotImplementedError

    def extract(self, dataset):
        """ Extracts features for each file in dataset.

        Call calculate() for each file in dataset and save the
        result into the features path.

        Parameters
        ----------
        dataset : Dataset
            Instance of the dataset.

        """
        features_path = self.get_features_path(dataset)
        mkdir_if_not_exists(features_path, parents=True)

        if not dataset.check_sampling_rate(self.sr):
            print('Changing sampling rate ...')
            dataset.change_sampling_rate(self.sr)
            print('Done!')

        # Define path to audio and features folders
        audio_path, subfolders = dataset.get_audio_paths(
            self.sr
        )

        # Duplicate folder structure of audio in features folder
        duplicate_folder_structure(audio_path, features_path)
        for audio_folder in subfolders:
            subfolder_name = os.path.basename(audio_folder)
            features_path_sub = os.path.join(features_path, subfolder_name)
            if not self.check_if_extracted_path(features_path_sub):
                # Navigate in the structure of audio folder and extract
                # features of the each wav file
                for path_audio in progressbar(list_wav_files(audio_folder)):
                    features_array = self.calculate(
                        path_audio
                    )
                    path_to_features_file = path_audio.replace(
                        audio_path, features_path
                    )
                    path_to_features_file = path_to_features_file.replace(
                        'wav', 'npy'
                    )
                    np.save(path_to_features_file, features_array)

                # Save parameters.json for future checking
                self.set_as_extracted(features_path_sub)

    def set_as_extracted(self, path):
        """ Saves a json file with self.__dict__.

        Useful for checking if the features files were calculated
        with same parameters.

        Parameters
        ----------
        path : str
            Path to the JSON file

        """
        params = self.__dict__.copy()
        remove = [
            key for key in params.keys() if type(params[key]) not in [
                int, str, float]
        ]
        for key in remove:
            del params[key]

        json_path = os.path.join(path, "parameters.json")
        with open(json_path, 'w') as fp:
            json.dump(params, fp)

    def check_if_extracted_path(self, path):
        """ Checks if the features saved in path were calculated.

        Compare if the features were calculated with the same parameters
        of self.__dict__.

        Parameters
        ----------
        path : str
            Path to the features folder

        Returns
        -------
        bool
            True if the features were already extracted.

        """
        json_features_folder = os.path.join(path, "parameters.json")
        if not os.path.exists(json_features_folder):
            return False
        parameters_features_folder = load_json(json_features_folder)
        for key in parameters_features_folder.keys():
            if key not in self.__dict__:
                return False
            if parameters_features_folder[key] != self.__dict__[key]:
                return False
        return True

    def check_if_extracted(self, dataset):
        """ Checks if the features of each file in dataset was calculated.

        Calls check_if_extracted_path for each path in the dataset.

        Parameters
        ----------
        path : str
            Path to the features folder

        Returns
        -------
        bool
            True if the features were already extracted.

        """
        features_path = self.get_features_path(dataset)
        audio_path, subfolders = dataset.get_audio_paths(self.sr)
        for audio_folder in subfolders:
            subfolder_name = os.path.basename(audio_folder)
            features_path_sub = os.path.join(features_path, subfolder_name)
            feat_extracted = self.check_if_extracted_path(features_path_sub)
            if not feat_extracted:
                return False

        return True

    def get_shape(self, length_sec=10.0):
        """
        Calls calculate() with a dummy signal of length length_sec
        and returns the shape of the feature representation.

        Parameters
        ----------
        length_sec : float
            Duration in seconds of the test signal

        Returns
        -------
        tuple
            Shape of the feature representation
        """

        audio_sample = np.zeros(int(length_sec*self.sr))
        audio_file = 'zeros.wav'
        sf.write('zeros.wav', audio_sample, self.sr)
        features_sample = self.calculate(audio_file)
        os.remove(audio_file)
        return features_sample.shape

    def get_features_path(self, dataset):
        """ Returns the path to the features folder.

        Parameters
        ----------
        dataset : Dataset
            Instance of the dataset.

        Returns
        -------
        features_path : str
            Path to the features folder.

        """
        feature_name = self.__class__.__name__
        features_path = os.path.join(
            dataset.dataset_path, self.features_folder, feature_name
        )
        return features_path

    def pad_audio(self, audio):
        if (self.sequence_time > 0) & (self.pad_mode is not None):
            sequence_samples = self.sequence_frames*self.audio_hop + self.audio_win
            sequence_hop_samples = self.sequence_hop*self.audio_hop
            if len(audio) < sequence_samples:
                audio = librosa.util.fix_length(
                    audio, sequence_samples, axis=0, mode=self.pad_mode)
            else:
                if self.sequence_hop_time > 0:
                    audio_frames = int((len(audio) - self.audio_win) / self.audio_hop) + int(((len(audio) - self.audio_win) % self.audio_hop)>0)
                    n_sequences = int((audio_frames - self.sequence_frames)/self.sequence_hop) + int(((audio_frames - self.sequence_frames) % self.sequence_hop)>0)
                    new_frames = n_sequences*self.sequence_hop + self.sequence_frames
                    new_samples = new_frames * self.audio_hop + self.audio_win
                    audio = librosa.util.fix_length(
                        audio,
                        new_samples,
                        axis=0, mode=self.pad_mode
                    )
                else:
                    audio = audio[:sequence_samples]

        return audio

    def convert_to_sequences(self, audio_representation):
        if (self.sequence_time > 0) & (self.sequence_hop_time > 0):
            audio_representation = np.ascontiguousarray(audio_representation)
            audio_representation = librosa.util.frame(
                audio_representation,
                self.sequence_frames,
                self.sequence_hop,
                axis=0
            )
        else:
            audio_representation = np.expand_dims(audio_representation, axis=0)
            if self.sequence_time > 0:
                audio_representation = audio_representation[:,:self.sequence_frames]

        return audio_representation