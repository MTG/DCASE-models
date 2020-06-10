import os
import numpy as np
import librosa
import openl3
import inspect
import sys

from .feature_extractor import FeatureExtractor

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


def get_available_features():
    available_features = {m[0]:m[1] for m in inspect.getmembers(
        sys.modules[__name__], inspect.isclass) if m[1].__module__ == __name__}

    return available_features