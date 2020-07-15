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

    Extract the log-scaled spectrogram of the audio signals. The spectrogram
    is calculated over the whole audio signal and then is separated in
    overlapped sequences (frames)

    Notes
    -----
    Based in librosa.core.stft function.

    Parameters
    ----------
    n_fft : int, default=1024
        Number of samples used for FFT calculation. Refer to librosa.core.stft
        for further information.

    pad_mode : str or None, default='reflect'
        Mode of padding applied to the audio signal. This argument is passed
        to librosa.util.fix_length for padding the signal. If pad_mode is None,
        no padding is applied.

    See Also
    --------
    FeatureExtractor : FeatureExtractor base class

    MelSpectrogram : MelSpectrogram feature extractor

    Examples
    --------
    Extract features of a given file

    >>> from dcase_models.data.features import Spectrogram
    >>> from dcase_models.util.files import example_audio_file
    >>> features = Spectrogram()
    >>> features_shape = features.get_shape()
    >>> print(features_shape)
        (21, 32, 513)
    >>> file_name = example_audio_file()
    >>> spectrogram = features.calculate(file_name)
    >>> print(spectrogram.shape)
        (3, 32, 513)

    Extract features for each file in a given dataset

    >>> from dcase_models.data.datasets import ESC50
    >>> dataset = ESC50('../datasets/ESC50')
    >>> features.extract(dataset)
    """

    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=680, sr=22050,
                 n_fft=1024, pad_mode='reflect'):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.n_fft = n_fft
        self.pad_mode = pad_mode

    def calculate(self, file_name):
        audio = self.load_audio(file_name)

        # Padding
        if self.pad_mode is not None:
            audio = librosa.util.fix_length(
                audio,
                audio.shape[0] + librosa.core.frames_to_samples(
                    self.sequence_frames, self.audio_hop, n_fft=self.n_fft),
                axis=0, mode=self.pad_mode
            )

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

        # Convert to sequences (frames),
        # shape (N_sequences, N_sequence_frames, N_freqs)
        spectrogram = np.ascontiguousarray(spectrogram)
        spectrogram = librosa.util.frame(
            spectrogram, self.sequence_frames, self.sequence_hop, axis=0
        )

        return spectrogram


class MelSpectrogram(FeatureExtractor):
    """ MelSpectrogram feature extractor.

    Extract the log-scaled mel-spectrogram of the audio signals.
    The mel-spectrogram is calculated over the whole audio signal and then is
    separated in overlapped sequences (frames)

    Notes
    -----
    Based in `librosa.core.stft` and `librosa.filters.mel` functions.

    Parameters
    ----------
    n_fft : int, default=1024
        Number of samples used for FFT calculation.
        Refer to `librosa.core.stft` for further information.

    mel_bands : int, default=64
        Number of mel bands.

    pad_mode : str or None, default='reflect'
        Mode of padding applied to the audio signal. This argument is passed
        to librosa.util.fix_length for padding the signal. If pad_mode is None,
        no padding is applied.

    kwargs
        Additional keyword arguments to `librosa.filters.mel`.


    See Also
    --------
    FeatureExtractor : FeatureExtractor base class

    Spectrogram : Spectrogram features


    Examples
    --------
    Extract features of a given file

    >>> from dcase_models.data.features import MelSpectrogram
    >>> from dcase_models.util.files import example_audio_file
    >>> features = MelSpectrogram()
    >>> features_shape = features.get_shape()
    >>> print(features_shape)
        (21, 32, 64)
    >>> file_name = example_audio_file()
    >>> mel_spectrogram = features.calculate(file_name)
    >>> print(mel_spectrogram.shape)
        (3, 32, 64)

    Extract features for each file in a given dataset

    >>> from dcase_models.data.datasets import ESC50
    >>> dataset = ESC50('../datasets/ESC50')
    >>> features.extract(dataset)

    """
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=680, sr=22050,
                 n_fft=1024, mel_bands=64,
                 pad_mode='reflect', **kwargs):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.n_fft = n_fft
        self.pad_mode = pad_mode
        self.mel_bands = mel_bands

        kwargs.setdefault('htk', True)
        kwargs.setdefault('fmax', None)

        self.mel_basis = librosa.filters.mel(
            sr, n_fft, mel_bands, **kwargs)

    def calculate(self, file_name):
        # Load audio
        audio = self.load_audio(file_name)
        # if len(audio) < self.audio_win:
        #     return None

        # Pad audio signal
        if self.pad_mode is not None:
            audio = librosa.util.fix_length(
                audio,
                audio.shape[0] + librosa.core.frames_to_samples(
                    self.sequence_frames, self.audio_hop, n_fft=self.n_fft),
                axis=0, mode=self.pad_mode
            )

        # Get the spectrogram, shape (N_freqs, N_frames)
        stft = librosa.core.stft(audio, n_fft=self.n_fft,
                                 hop_length=self.audio_hop,
                                 win_length=self.audio_win, center=False)
        # Convert to power
        spectrogram = np.abs(stft)**2

        # Convert to mel_spectrogram, shape (N_bands, N_frames)
        mel_spectrogram = self.mel_basis.dot(spectrogram)
        assert mel_spectrogram.shape[0] == self.mel_bands

        # Convert to db
        mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # Transpose time and freq dims, shape (N_frames, N_bands)
        mel_spectrogram = mel_spectrogram.T

        # Pad the mel_spectrogram, shape (N_frames', N_bands)
        # mel_spectrogram = librosa.util.fix_length(
        #     mel_spectrogram,
        #     mel_spectrogram.shape[0]+self.sequence_frames,
        #     axis=0, mode='reflect'
        # )

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

    Parameters
    ----------
    content_type : {'music' or 'env'}, default='env'
        Type of content used to train the embedding model.
        Refer to openl3.core.get_audio_embedding.

    input_repr : {'linear', 'mel128', or 'mel256'}
        Spectrogram representation used for model.
        Refer to openl3.core.get_audio_embedding.

    embedding_size : {6144 or 512}, default=512
        Embedding dimensionality.
        Refer to openl3.core.get_audio_embedding.

    pad_mode : str or None, default='reflect'
        Mode of padding applied to the audio signal. This argument is passed
        to librosa.util.fix_length for padding the signal. If pad_mode is None,
        no padding is applied.

    See Also
    --------
    FeatureExtractor : FeatureExtractor base class

    Spectrogram : Spectrogram features


    Examples
    --------
    Extract features of a given file

    >>> from dcase_models.data.features import Openl3
    >>> from dcase_models.util.files import example_audio_file
    >>> features = Openl3()
    >>> features_shape = features.get_shape()
    >>> print(features_shape)
        (20, 512)
    >>> file_name = example_audio_file()
    >>> mel_spectrogram = features.calculate(file_name)
    >>> print(mel_spectrogram.shape)
        (3, 512)

    Extract features for each file in a given dataset

    >>> from dcase_models.data.datasets import ESC50
    >>> dataset = ESC50('../datasets/ESC50')
    >>> features.extract(dataset)

    """
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=680, sr=22050,
                 content_type="env", input_repr="mel256", embedding_size=512):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.content_type = content_type
        self.input_repr = input_repr
        self.embedding_size = embedding_size

    def calculate(self, file_name):
        audio = self.load_audio(file_name, change_sampling_rate=False)
        emb, ts = openl3.get_audio_embedding(
            audio, self.sr,
            content_type=self.content_type,
            embedding_size=self.embedding_size,
            input_repr=self.input_repr,
            hop_size=self.sequence_hop_time,
            verbose=False
        )

        return emb


class RawAudio(FeatureExtractor):
    """ RawAudio feature extractor.

    Load the audio signal and create sequences (overlapped windows)

    Parameters
    ----------
    pad_mode : str or None, default='reflect'
        Mode of padding applied to the audio signal. This argument is passed
        to librosa.util.fix_length for padding the signal. If pad_mode is None,
        no padding is applied.

    """
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=680, sr=22050,
                 pad_mode='reflect'):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.pad_mode = pad_mode

        self.sequence_samples = int(librosa.core.frames_to_samples(
            self.sequence_frames, audio_hop))
        self.sequence_hop_samples = int(librosa.core.frames_to_samples(
            self.sequence_hop, audio_hop))

    def calculate(self, file_name):
        audio = self.load_audio(file_name, change_sampling_rate=False)

        if self.pad_mode is not None:
            audio = librosa.util.fix_length(
                audio,
                audio.shape[0] + self.sequence_samples,
                axis=0, mode=self.pad_mode
            )

        audio = np.ascontiguousarray(audio)
        audio_seqs = librosa.util.frame(
            audio, self.sequence_samples, self.sequence_hop_samples, axis=0
        )

        return audio_seqs


class FramesAudio(FeatureExtractor):
    """ FramesAudio feature extractor.

    Load the audio signal, convert it into time-short frames, and create
    sequences (overlapped windows).

    Parameters
    ----------
    pad_mode : str or None, default='reflect'
        Mode of padding applied to the audio signal. This argument is passed
        to librosa.util.fix_length for padding the signal. If pad_mode is None,
        no padding is applied.

    """
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=680, sr=22050, n_fft=1024,
                 pad_mode='reflect'):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.n_fft = n_fft
        self.pad_mode = pad_mode

        self.sequence_samples = librosa.core.frames_to_samples(
            self.sequence_frames, audio_hop, n_fft)
        self.sequence_hop_samples = librosa.core.frames_to_samples(
            self.sequence_hop, audio_hop, n_fft)

    def calculate(self, file_name):
        audio = self.load_audio(file_name, change_sampling_rate=False)

        if self.pad_mode is not None:
            audio = librosa.util.fix_length(
                audio,
                audio.shape[0] + self.sequence_samples,
                axis=0, mode=self.pad_mode
            )

        audio = np.ascontiguousarray(audio)
        audio_frames = librosa.util.frame(
            audio, self.audio_win, self.audio_hop, axis=0
        )
        # TODO: ADD WINDOWING

        audio_frames = np.ascontiguousarray(audio_frames)
        audio_seqs = librosa.util.frame(
            audio_frames, self.sequence_frames, self.sequence_hop, axis=0
        )

        return audio_seqs


def get_available_features():
    available_features = {m[0]: m[1] for m in inspect.getmembers(
        sys.modules[__name__], inspect.isclass) if m[1].__module__ == __name__}

    return available_features
