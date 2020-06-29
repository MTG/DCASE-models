import numpy as np
import librosa
import openl3
import inspect
import sys

from .feature_extractor import FeatureExtractor


__all__ = ['Spectrogram', 'MelSpectrogram',
           'Openl3', 'RawAudio', 'FramesAudio']


class Spectrogram(FeatureExtractor):
    """ Spectrogram feature extractor.

    Based in librosa.core.stft function.

    Note:
    -----
    The spectrogram is calculated all over the file and then
    it is separated in overlapped frames.

    """

    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=512, sr=44100, n_fft=1024):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.n_fft = n_fft
        self.params['name'] = 'Spectrogram'
        self.params['n_fft'] = 'n_fft'

    def calculate(self, file_name):
        audio = self.load_audio(file_name)

        # Spectrogram, shape (N_frames, N_freqs)
        stft = librosa.core.stft(audio, n_fft=self.n_fft,
                                 hop_length=self.audio_hop,
                                 win_length=self.audio_win, center=True)

        # Power
        spectrogram = np.abs(stft)**2

        # Convert to db
        spectrogram = librosa.power_to_db(spectrogram)

        # Transpose time and freq dims, shape
        spectrogram = spectrogram.T

        # Pad the spectrogram, shape (N_frames', N_freqs)
        spectrogram = librosa.util.fix_length(
            spectrogram,
            spectrogram.shape[0]+self.sequence_frames,
            axis=0, mode='reflect'
        )

        # Convert to sequences (frames),
        # shape (N_sequences, N_sequence_frames, N_freqs)
        spectrogram = np.ascontiguousarray(spectrogram)
        spectrogram = librosa.util.frame(
            spectrogram, self.sequence_frames, self.sequence_hop, axis=0
        )

        return spectrogram


class MelSpectrogram(FeatureExtractor):
    """ MelSpectrogram feature extractor.

    Based in librosa.core.stft and librosa.filters.mel.

    Note:
    -----
    The mel-spectogram is calculated all over the file and then
    it is separated in overlapped frames.

    """

    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=512, sr=44100,
                 n_fft=1024, mel_bands=128, fmax=None):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.params['name'] = 'MelSpectrogram'
        self.params['mel_bands'] = mel_bands
        self.params['fmax'] = fmax
        self.params['n_fft'] = 'n_fft'

        self.n_fft = n_fft
        self.mel_basis = librosa.filters.mel(
            sr, n_fft, mel_bands, htk=True, fmax=fmax)

    def calculate(self, file_name):
        # Load audio
        audio = self.load_audio(file_name)
        # if len(audio) < self.audio_win:
        #     return None

        # Get the spectrogram, shape (N_freqs, N_frames)
        stft = librosa.core.stft(audio, n_fft=self.n_fft,
                                 hop_length=self.audio_hop,
                                 win_length=self.audio_win, center=False)
        # Convert to power
        spectrogram = np.abs(stft)**2

        # Convert to mel_spectrogram, shape (N_bands, N_frames)
        mel_spectrogram = self.mel_basis.dot(spectrogram)
        assert mel_spectrogram.shape[0] == self.params['mel_bands']

        # Convert to db
        mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # Transpose time and freq dims, shape (N_frames, N_bands)
        mel_spectrogram = mel_spectrogram.T

        # Pad the mel_spectrogram, shape (N_frames', N_bands)
        mel_spectrogram = librosa.util.fix_length(
            mel_spectrogram,
            mel_spectrogram.shape[0]+self.sequence_frames,
            axis=0, mode='reflect'
        )

        # Convert to sequences (frames),
        # shape (N_sequences, N_sequence_frames, N_bands)
        mel_spectrogram = np.ascontiguousarray(mel_spectrogram)
        mel_spectrogram = librosa.util.frame(
            mel_spectrogram, self.sequence_frames, self.sequence_hop, axis=0
        )

        return mel_spectrogram


class Openl3(FeatureExtractor):
    """ Openl3 feature extractor.

    Based in openl3 library.

    """
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=512, sr=44100,
                 content_type="env", input_repr="mel256", embedding_size=512):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.params['name'] = 'Openl3'
        self.params['content_type'] = content_type
        self.params['input_repr'] = input_repr
        self.params['embedding_size'] = embedding_size

    def calculate(self, file_name):
        audio = self.load_audio(file_name, change_sampling_rate=False)
        emb, ts = openl3.get_audio_embedding(
            audio, self.sr,
            content_type=self.params['content_type'],
            embedding_size=self.params['embedding_size'],
            input_repr=self.params['input_repr'],
            hop_size=self.sequence_hop_time,
            verbose=False
        )

        return emb


class RawAudio(FeatureExtractor):
    """ RawAudio feature extractor.

    Only load audio and create sequences (overlapped windows)

    """
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=512, sr=44100):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.sequence_samples = audio_hop * self.sequence_frames
        self.sequence_hop_samples = audio_hop * self.sequence_hop

    def calculate(self, file_name):
        audio = self.load_audio(file_name, change_sampling_rate=False)

        audio = librosa.util.fix_length(
            audio,
            audio.shape[0]+self.sequence_samples,
            axis=0, mode='constant'
        )

        audio = np.ascontiguousarray(audio)
        audio_seqs = librosa.util.frame(
            audio, self.sequence_samples, self.sequence_hop_samples, axis=0
        )

        return audio_seqs


class FramesAudio(FeatureExtractor):
    """ FramesAudio feature extractor.

    Only load audio and create sequences (overlapped windows)

    """
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=512, sr=44100):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.sequence_samples = audio_hop * self.sequence_frames
        self.sequence_hop_samples = audio_hop * self.sequence_hop

    def calculate(self, file_name):
        audio = self.load_audio(file_name, change_sampling_rate=False)

        audio = np.ascontiguousarray(audio)
        audio_frames = librosa.util.frame(
            audio, self.audio_win, self.audio_hop, axis=0
        )

        audio_frames = librosa.util.fix_length(
            audio_frames,
            audio_frames.shape[0]+self.sequence_frames,
            axis=0, mode='constant'
        )

        audio_seqs = librosa.util.frame(
            audio_frames, self.sequence_frames, self.sequence_hop, axis=0
        )

        return audio_seqs


def get_available_features():
    available_features = {m[0]: m[1] for m in inspect.getmembers(
        sys.modules[__name__], inspect.isclass) if m[1].__module__ == __name__}

    return available_features
