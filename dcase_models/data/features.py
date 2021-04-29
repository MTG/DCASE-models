import numpy as np
import librosa
import inspect
import sys

from dcase_models.data.feature_extractor import FeatureExtractor


__all__ = ['Spectrogram', 'MelSpectrogram', 'MFCC',
           'Openl3', 'RawAudio', 'FramesAudio',
           'VGGishEmbeddings']


class Spectrogram(FeatureExtractor):
    """ Spectrogram feature extractor.

    Extracts the log-scaled spectrogram of the audio signals. The spectrogram
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
    FeatureExtractor : FeatureExtractor base class.

    MelSpectrogram : MelSpectrogram feature extractor.

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

    Extract features for each file in a given dataset.

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
        audio = self.pad_audio(audio)

        # Spectrogram, shape (N_frames, N_freqs)
        stft = librosa.core.stft(audio, n_fft=self.n_fft,
                                 hop_length=self.audio_hop,
                                 win_length=self.audio_win, center=False)

        # Power
        spectrogram = np.abs(stft)**2

        # Convert to db
        spectrogram = librosa.power_to_db(spectrogram)

        # Transpose time and freq dims, shape
        spectrogram = spectrogram.T

        # Convert to sequences (frames),
        # shape (N_sequences, N_sequence_frames, N_freqs)
        # spectrogram = np.ascontiguousarray(spectrogram)
        # spectrogram = librosa.util.frame(
        #     spectrogram, self.sequence_frames, self.sequence_hop, axis=0
        # )
        spectrogram = self.convert_to_sequences(spectrogram)

        return spectrogram


class MelSpectrogram(FeatureExtractor):
    """ MelSpectrogram feature extractor.

    Extracts the log-scaled mel-spectrogram of the audio signals.
    The mel-spectrogram is calculated over the whole audio signal and then is
    separated in overlapped sequences (frames).

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
    Extract features of a given file.

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

    Extract features for each file in a given dataset.

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
        audio = self.pad_audio(audio)
                        
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
        mel_spectrogram = self.convert_to_sequences(mel_spectrogram)

        return mel_spectrogram


class MFCC(FeatureExtractor):
    """ MFCC feature extractor.

    Extracts Mel-frequency cepstral coefficients (MFCCs).
    The MFCCS are calculated over the whole audio signal and then are
    separated in overlapped sequences (frames).

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
    Extract features of a given file.

    >>> from dcase_models.data.features import MFCC
    >>> from dcase_models.util.files import example_audio_file
    >>> features = MFCC()
    >>> features_shape = features.get_shape()
    >>> print(features_shape)
        (21, 32, 20)
    >>> file_name = example_audio_file()
    >>> mfcc = features.calculate(file_name)
    >>> print(mfcc.shape)
        (3, 32, 20)

    Extract features for each file in a given dataset.

    >>> from dcase_models.data.datasets import ESC50
    >>> dataset = ESC50('../datasets/ESC50')
    >>> features.extract(dataset)

    """
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=680, sr=22050,
                 n_fft=1024, n_mfcc=20, dct_type=2,
                 norm='ortho', lifter=0,
                 pad_mode='reflect', **kwargs):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         sr=sr)

        self.n_fft = n_fft
        self.pad_mode = pad_mode
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.lifter = lifter

        kwargs.setdefault('htk', True)
        kwargs.setdefault('fmax', None)
        kwargs.setdefault('norm', 1)
        kwargs.setdefault('fmin', 0.0)
        kwargs.setdefault('fmax', 0.0)
        kwargs.setdefault('n_mels', 128)

        self.mel_basis = librosa.filters.mel(
            sr, n_fft, **kwargs)

    def calculate(self, file_name):
        # Load audio
        audio = self.load_audio(file_name)
        # if len(audio) < self.audio_win:
        #     return None

        # Pad audio signal
        audio = self.pad_audio(audio)

        # Get the spectrogram, shape (N_freqs, N_frames)
        stft = librosa.core.stft(audio, n_fft=self.n_fft,
                                 hop_length=self.audio_hop,
                                 win_length=self.audio_win, center=False)
        # Convert to power
        spectrogram = np.abs(stft)**2

        # Convert to mel_spectrogram, shape (N_bands, N_frames)
        mel_spectrogram = self.mel_basis.dot(spectrogram)

        # Convert to db
        mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # Calculate MFCCs
        mfcc = librosa.feature.mfcc(S=mel_spectrogram,
                                    n_mfcc=self.n_mfcc,
                                    dct_type=self.dct_type,
                                    norm=self.norm,
                                    lifter=self.lifter)

        assert mfcc.shape[0] == self.n_mfcc

        # Transpose time and freq dims, shape (N_frames, N_MFCC)
        mfcc = mfcc.T

        # Convert to sequences (frames),
        # shape (N_sequences, N_sequence_frames, N_MFCC)
        # mfcc = np.ascontiguousarray(mfcc)
        # mfcc = librosa.util.frame(
        #     mfcc, self.sequence_frames, self.sequence_hop, axis=0
        # )
        mfcc = self.convert_to_sequences(mfcc)

        return mfcc


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
    Extract features of a given file.

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

    Extract features for each file in a given dataset.

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
        import tensorflow as tf
        tensorflow2 = tf.__version__.split('.')[0] == '2'
        if tensorflow2:
            raise ImportError("Openl3 requires tensorflow1")
        import openl3
        self.content_type = content_type
        self.input_repr = input_repr
        self.embedding_size = embedding_size
        self.openl3 = openl3.models.load_audio_embedding_model(
            input_repr, content_type, embedding_size)

    def calculate(self, file_name):
        import openl3
        audio = self.load_audio(file_name, change_sampling_rate=False)
        emb, ts = openl3.get_audio_embedding(
            audio, self.sr,
            model=self.openl3,
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
        audio = self.pad_audio(audio)

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

        audio = self.pad_audio(audio)

        audio = np.ascontiguousarray(audio)
        audio_frames = librosa.util.frame(
            audio, self.audio_win, self.audio_hop, axis=0
        )
        # TODO: ADD WINDOWING

        # audio_frames = np.ascontiguousarray(audio_frames)
        # audio_seqs = librosa.util.frame(
        #     audio_frames, self.sequence_frames, self.sequence_hop, axis=0
        # )

        audio_seqs = self.convert_to_sequences(audio_frames)

        return audio_seqs


class VGGishEmbeddings(FeatureExtractor):
    """ VGGish embeddings feature extractor.

    Extract embeddings from VGGish model.

    Parameters
    ----------
    pad_mode : str or None, default='reflect'
        Mode of padding applied to the audio signal. This argument is passed
        to librosa.util.fix_length for padding the signal. If pad_mode is None,
        no padding is applied.

    """
    def __init__(self, sequence_hop_time=0.96,
                 pad_mode='reflect', include_top=True, compress=True):

        from dcase_models.model.models import VGGish
        
        sequence_time = 0.96
        audio_win = 400
        audio_hop = 160
        sr = 16000
        n_fft = 512
        self.mel_bands = 64
        self.fmin = 150
        self.fmax = 7500

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

        self.vggish = VGGish(
            model=None, model_path=None, metrics=[],
            n_frames_cnn=96, n_freq_cnn=64, n_classes=0,
            embedding_size=128, pooling='avg', include_top=include_top, compress=compress)
        
        self.vggish.load_pretrained_model_weights()

    def frame(self, data, window_length, hop_length):
        """Convert array into a sequence of successive possibly overlapping frames.
        An n-dimensional array of shape (num_samples, ...) is converted into an
        (n+1)-D array of shape (num_frames, window_length, ...), where each
        frame starts hop_length points after the preceding one.
        This is accomplished using stride_tricks, so the original data is not
        copied.  However, there is no zero-padding, so any incomplete frames
        at the end are not included.
        Args:
            data: np.array of dimension N >= 1.
            window_length: Number of samples in each frame.
            hop_length: Advance (in samples) between each window.
        Returns:
            (N+1)-D np.array with as many rows as there are complete frames
            that can be extracted.
        """
        num_samples = data.shape[0]
        num_frames = 1 + int(
            np.floor((num_samples - window_length) / hop_length))
        shape = (num_frames, window_length) + data.shape[1:]
        strides = (data.strides[0] * hop_length,) + data.strides
        return np.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides)

    def periodic_hann(self, window_length):
        """Calculate a "periodic" Hann window.
        The classic Hann window is defined as a raised cosine that starts and
        ends on zero, and where every value appears twice, except the middle
        point for an odd-length window.  Matlab calls this a "symmetric" window
        and np.hanning() returns it.  However, for Fourier analysis, this
        actually represents just over one cycle of a period N-1 cosine, and
        thus is not compactly expressed on a length-N Fourier basis.  Instead,
        it's better to use a raised cosine that ends just before the final
        zero value - i.e. a complete cycle of a period-N cosine.  Matlab
        calls this a "periodic" window. This routine calculates it.
        Args:
            window_length: The number of points in the returned window.
        Returns:
            A 1D np.array containing the periodic hann window.
        """
        return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                                    np.arange(window_length)))

    def stft_magnitude(self, signal, fft_length,
                       hop_length=None,
                       window_length=None):
        """Calculate the short-time Fourier transform magnitude.
        Args:
            signal: 1D np.array of the input time-domain signal.
            fft_length: Size of the FFT to apply.
            hop_length: Advance (in samples) between each frame passed to FFT.
            window_length: Length of each block of samples to pass to FFT.
        Returns:
            2D np.array where each row contains the magnitudes of the
            fft_length/2+1 unique values of the FFT for the corresponding
            frame of input samples.
        """
        frames = self.frame(signal, window_length, hop_length)
        # Apply frame window to each frame. We use a periodic Hann
        # (cosine of period window_length) instead of the symmetric Hann of
        # np.hanning (period window_length-1).
        window = self.periodic_hann(window_length)
        windowed_frames = frames * window
        return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))

    def hertz_to_mel(self, frequencies_hertz):
        """Convert frequencies to mel scale using HTK formula.
        Args:
            frequencies_hertz: Scalar or np.array of frequencies in hertz.
        Returns:
            Object of same size as frequencies_hertz containing corresponding
            values on the mel scale.
        """
        # Mel spectrum constants and functions.
        _MEL_BREAK_FREQUENCY_HERTZ = 700.0
        _MEL_HIGH_FREQUENCY_Q = 1127.0
        return _MEL_HIGH_FREQUENCY_Q * np.log(
            1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

    def spectrogram_to_mel_matrix(self, num_mel_bins=20,
                                  num_spectrogram_bins=129,
                                  audio_sample_rate=8000,
                                  lower_edge_hertz=125.0,
                                  upper_edge_hertz=3800.0):
        """Return a matrix that can post-multiply spectrogram rows to make mel.
        Returns a np.array matrix A that can be used to post-multiply a matrix
        S of spectrogram values (STFT magnitudes) arranged as frames x bins to
        generate a "mel spectrogram" M of frames x num_mel_bins.  M = S A.
        The classic HTK algorithm exploits the complementarity of adjacent mel
        bands to multiply each FFT bin by only one mel weight, then add it,
        with positive and negative signs, to the two adjacent mel bands to
        which that bin contributes.  Here, by expressing this operation as a
        matrix multiply, we go from num_fft multiplies per frame
        (plus around 2*num_fft adds) to around num_fft^2 multiplies and adds.
        However, because these are all presumably accomplished in a single
        call to np.dot(), it's not clear which approach is faster in Python.
        The matrix multiplication has the attraction of being more
        general and flexible, and much easier to read.
        Args:
            num_mel_bins: How many bands in the resulting mel spectrum.  This is
            the number of columns in the output matrix.
            num_spectrogram_bins: How many bins there are in the source spectrogram
            data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
            only contains the nonredundant FFT bins.
            audio_sample_rate: Samples per second of the audio at the input to the
            spectrogram. We need this to figure out the actual frequencies for
            each spectrogram bin, which dictates how they are mapped into mel.
            lower_edge_hertz: Lower bound on the frequencies to be included in the mel
            spectrum.  This corresponds to the lower edge of the lowest triangular
            band.
            upper_edge_hertz: The desired top edge of the highest frequency band.
        Returns:
            An np.array with shape (num_spectrogram_bins, num_mel_bins).
        Raises:
            ValueError: if frequency edges are incorrectly ordered.
        """
        nyquist_hertz = audio_sample_rate / 2.
        if lower_edge_hertz >= upper_edge_hertz:
            raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                            (lower_edge_hertz, upper_edge_hertz))
        spectrogram_bins_hertz = np.linspace(
            0.0, nyquist_hertz, num_spectrogram_bins)
        spectrogram_bins_mel = self.hertz_to_mel(spectrogram_bins_hertz)
        # The i'th mel band (starting from i=1) has center frequency
        # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
        # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
        # the band_edges_mel arrays.
        band_edges_mel = np.linspace(self.hertz_to_mel(lower_edge_hertz),
                                     self.hertz_to_mel(upper_edge_hertz),
                                     num_mel_bins + 2)
        # Matrix to post-multiply feature arrays whose rows are 
        # num_spectrogram_bins of spectrogram values.
        mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
        for i in range(num_mel_bins):
            lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
            # Calculate lower and upper slopes for every spectrogram bin.
            # Line segments are linear in the *mel* domain, not hertz.
            lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                           (center_mel - lower_edge_mel))
            upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                           (upper_edge_mel - center_mel))
            # .. then intersect them with each other and zero.
            mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                                  upper_slope))
        # HTK excludes the spectrogram DC bin; make sure it always gets a zero
        # coefficient.
        mel_weights_matrix[0, :] = 0.0
        return mel_weights_matrix

    def log_mel_spectrogram(self, data,
                            audio_sample_rate=8000,
                            log_offset=0.0,
                            window_length_secs=0.025,
                            hop_length_secs=0.010,
                            **kwargs):
        """Convert waveform to a log magnitude mel-frequency spectrogram.
        Args:
            data: 1D np.array of waveform data.
            audio_sample_rate: The sampling rate of data.
            log_offset: Add this to values when taking log to avoid -Infs.
            window_length_secs: Duration of each window to analyze.
            hop_length_secs: Advance between successive analysis windows.
            **kwargs: Additional arguments to pass to
            spectrogram_to_mel_matrix.
        Returns:
            2D np.array of (num_frames, num_mel_bins) consisting of log mel
            filterbank magnitudes for successive frames.
        """
        window_length_samples = int(
            round(audio_sample_rate * window_length_secs))
        hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
        fft_length = 2 ** int(
            np.ceil(np.log(window_length_samples) / np.log(2.0)))
        spectrogram = self.stft_magnitude(
            data,
            fft_length=fft_length,
            hop_length=hop_length_samples,
            window_length=window_length_samples)
        mel_spectrogram = np.dot(spectrogram, self.spectrogram_to_mel_matrix(
            num_spectrogram_bins=spectrogram.shape[1],
            audio_sample_rate=audio_sample_rate, **kwargs))
        return np.log(mel_spectrogram + log_offset)

    def calculate(self, file_name):
        audio = self.load_audio(file_name, change_sampling_rate=False)

        if self.pad_mode is not None:
            audio = librosa.util.fix_length(
                audio,
                audio.shape[0] + self.sequence_samples,
                axis=0, mode=self.pad_mode
            )

        mel_spectrogram = self.log_mel_spectrogram(
            audio,
            audio_sample_rate=16000,
            log_offset=0.01,
            window_length_secs=0.025,
            hop_length_secs=0.010,
            num_mel_bins=64,
            lower_edge_hertz=150,
            upper_edge_hertz=7500
        )
        mel_spectrogram = np.ascontiguousarray(mel_spectrogram)
        mel_spectrogram = librosa.util.frame(
            mel_spectrogram, self.sequence_frames, self.sequence_hop, axis=0
        )

        emb = self.vggish.model.predict(mel_spectrogram)

        return emb


def get_available_features():
    available_features = {m[0]: m[1] for m in inspect.getmembers(
        sys.modules[__name__], inspect.isclass) if m[1].__module__ == __name__}

    return available_features
