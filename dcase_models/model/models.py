from functools import partial
import inspect
import sys
import os

import tensorflow as tf
tensorflow2 = tf.__version__.split('.')[0] == '2'

if tensorflow2:
    from tensorflow.keras.layers import GRU, Bidirectional
    from tensorflow.keras.layers import TimeDistributed, Activation, Reshape
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import GlobalMaxPooling2D
    from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Dropout, Dense, Flatten
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    import tensorflow.keras.backend as K
else:
    from keras.layers import GRU, Bidirectional
    from keras.layers import TimeDistributed, Activation, Reshape
    from keras.layers import GlobalAveragePooling2D
    from keras.layers import GlobalMaxPooling2D
    from keras.layers import Input, Lambda, Conv2D, MaxPooling2D
    from keras.layers import Conv1D
    from keras.layers import Dropout, Dense, Flatten
    from keras.layers import BatchNormalization
    from keras.layers import Layer
    from keras.models import Model
    from keras.regularizers import l2
    import keras.backend as K
    

from tensorflow import clip_by_value

from dcase_models.model.container import KerasModelContainer


__all__ = ['MLP', 'SB_CNN', 'SB_CNN_SED', 'A_CRNN',
           'VGGish', 'SMel', 'MST']


class MLP(KerasModelContainer):
    """ KerasModelContainer for a generic MLP model.

    Parameters
    ----------
    n_classes : int, default=10
        Number of classes (dimmension output).

    n_frames : int or None, default=64
        Length of the input (number of frames of each sequence).
        Use None to not use frame-level input and output. In this case the
        input has shape (None, n_freqs).

    n_freqs : int, default=12
        Number of frequency bins. The model's input has shape
        (n_frames, n_freqs).

    hidden_layers_size : list of int, default=[128, 64]
        Dimmension of each hidden layer. Note that the length of this list
        defines the number of hidden layers.

    dropout_rates : list of float, default=[0.5, 0.5]
        List of dropout rate use after each hidden layer. The length of this
        list must be equal to the length of hidden_layers_size. Use 0.0
        (or negative) to not use dropout.

    hidden_activation : str, default='relu'
        Activation for hidden layers.

    l2_reg : float, default=1e-5
        Weight of the l2 regularizers. Use 0.0 to not use regularization.

    final_activation : str, default='softmax'
        Activation of the last layer.

    temporal_integration : {'mean', 'sum', 'autopool'}, default='mean'
        Temporal integration operation used after last layer.

    kwargs
        Additional keyword arguments to `Dense layers`.


    Attributes
    ----------
    model : keras.models.Model
        Keras model.

    Examples
    --------
    >>> from dcase_models.model.models import MLP
    >>> model_container = MLP()
    >>> model_container.model.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 64, 12)            0
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 64, 128)           1664
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 64, 128)           0
    _________________________________________________________________
    time_distributed_2 (TimeDist (None, 64, 64)            8256
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 64, 64)            0
    _________________________________________________________________
    time_distributed_3 (TimeDist (None, 64, 10)            650
    _________________________________________________________________
    temporal_integration (Lambda (None, 10)                0
    =================================================================
    Total params: 10,570
    Trainable params: 10,570
    Non-trainable params: 0
    _________________________________________________________________

    """

    def __init__(self, model=None, model_path=None,
                 metrics=['classification'], n_classes=10,
                 n_frames=64, n_freqs=12,
                 hidden_layers_size=[128, 64],
                 dropout_rates=[0.5, 0.5], hidden_activation='relu',
                 l2_reg=1e-5, final_activation='softmax',
                 temporal_integration='mean', **kwargs):

        # self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.n_freqs = n_freqs
        self.hidden_layers_size = hidden_layers_size
        self.dropout_rates = dropout_rates
        self.l2_reg = l2_reg
        self.temporal_integration = temporal_integration
        self.use_time_distributed = n_frames is not None
        self.hidden_activation = hidden_activation
        self.final_activation = final_activation
        self.kwargs = kwargs

        super().__init__(model=model, model_path=model_path,
                         model_name='MLP', metrics=metrics)

    def build(self):
        """ Missing docstring here
        """
        # input
        if self.use_time_distributed:
            input_shape = (self.n_frames, self.n_freqs)
        else:
            input_shape = (self.n_freqs,)

        inputs = Input(shape=input_shape, dtype='float32', name='input')
        y = inputs
        # Hidden layers
        for idx in range(len(self.hidden_layers_size)):
            dense_layer = Dense(self.hidden_layers_size[idx],
                                activation=self.hidden_activation,
                                kernel_regularizer=l2(self.l2_reg),
                                name='dense_{}'.format(idx+1), **self.kwargs)
            if self.use_time_distributed:
                y = TimeDistributed(dense_layer)(y)
            else:
                y = dense_layer(y)

            # Dropout
            if self.dropout_rates[idx] > 0:
                y = Dropout(self.dropout_rates[idx])(y)
        # Output layer
        dense_layer = Dense(self.n_classes, activation=self.final_activation,
                            kernel_regularizer=l2(self.l2_reg),
                            name='output', **self.kwargs)

        if self.use_time_distributed:
            y = TimeDistributed(dense_layer)(y)
        else:
            y = dense_layer(y)

        # Temporal integration
        if self.use_time_distributed:
            if self.temporal_integration == 'mean':
                y = Lambda(lambda x: K.mean(x, 1), name='temporal_integration')(y)
            elif self.temporal_integration == 'sum':
                y = Lambda(lambda x: K.sum(x, 1), name='temporal_integration')(y)
            elif self.temporal_integration == 'autopool':
                try:
                    from autopool import AutoPool1D
                except:
                    raise ImportError("Autopool is not installed")
                y = AutoPool1D(axis=1, name='output')(y)

        # Create model
        self.model = Model(inputs=inputs, outputs=y, name='model')

        super().build()


class SB_CNN(KerasModelContainer):
    """ KerasModelContainer for SB_CNN model.

    J. Salamon and J. P. Bello.
    "Deep Convolutional Neural Networks and Data Augmentation
    For Environmental Sound Classification".
    IEEE Signal Processing Letters, 24(3), pages 279 - 283.
    2017.

    Notes
    -----
    Code based on Salamon's implementation
    https://github.com/justinsalamon/scaper_waspaa2017


    Parameters
    ----------
    n_classes : int, default=10
        Number of classes (dimmension output).

    n_frames_cnn : int or None, default=64
        Length of the input (number of frames of each sequence).

    n_freq_cnn : int, default=128
        Number of frequency bins. The model's input has shape
        (n_frames, n_freqs).

    filter_size_cnn : tuple, default=(5,5)
        Kernel dimmension for convolutional layers.

    pool_size_cnn : tuple, default=(2,2)
        Pooling dimmension for maxpooling layers.

    n_dense_cnn : int, default=64
        Dimmension of penultimate dense layer.

    n_channels : int, default=0
        Number of input channels

        0 : mono signals.
            Input shape = (n_frames_cnn, n_freq_cnn)
        1 : mono signals.
            Input shape = (n_frames_cnn, n_freq_cnn, 1)
        2 : stereo signals.
            Input shape = (n_frames_cnn, n_freq_cnn, 2)
        n > 2 : multi-representations.
            Input shape = (n_frames_cnn, n_freq_cnn, n_channels)


    Attributes
    ----------
    model : keras.models.Model
        Keras model.

    Examples
    --------
    >>> from dcase_models.model.models import SB_CNN
    >>> model_container = SB_CNN()
    >>> model_container.model.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 64, 128)           0
    _________________________________________________________________
    lambda (Lambda)              (None, 64, 128, 1)        0
    _________________________________________________________________
    conv1 (Conv2D)               (None, 60, 124, 24)       624
    _________________________________________________________________
    maxpool1 (MaxPooling2D)      (None, 30, 62, 24)        0
    _________________________________________________________________
    batchnorm1 (BatchNormalizati (None, 30, 62, 24)        96
    _________________________________________________________________
    conv2 (Conv2D)               (None, 26, 58, 48)        28848
    _________________________________________________________________
    maxpool2 (MaxPooling2D)      (None, 6, 29, 48)         0
    _________________________________________________________________
    batchnorm2 (BatchNormalizati (None, 6, 29, 48)         192
    _________________________________________________________________
    conv3 (Conv2D)               (None, 2, 25, 48)         57648
    _________________________________________________________________
    batchnorm3 (BatchNormalizati (None, 2, 25, 48)         192
    _________________________________________________________________
    flatten (Flatten)            (None, 2400)              0
    _________________________________________________________________
    dropout1 (Dropout)           (None, 2400)              0
    _________________________________________________________________
    dense1 (Dense)               (None, 64)                153664
    _________________________________________________________________
    dropout2 (Dropout)           (None, 64)                0
    _________________________________________________________________
    out (Dense)                  (None, 10)                650
    =================================================================
    Total params: 241,914
    Trainable params: 241,674
    Non-trainable params: 240
    _________________________________________________________________
    """

    def __init__(self, model=None, model_path=None, metrics=['classification'],
                 n_classes=10, n_frames_cnn=64,
                 n_freq_cnn=128, filter_size_cnn=(5, 5), pool_size_cnn=(2, 2),
                 n_dense_cnn=64, n_channels=0):
        """ Initialization of the SB-CNN model.

        """
        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn
        self.n_freq_cnn = n_freq_cnn
        self.filter_size_cnn = filter_size_cnn
        self.pool_size_cnn = pool_size_cnn
        self.n_dense_cnn = n_dense_cnn
        self.n_channels = n_channels

        super().__init__(
            model=model, model_path=model_path,
            model_name='SB_CNN', metrics=metrics
        )

    def build(self):
        """ Builds the CNN Keras model according to the initialized parameters.
        """
        # Here define the keras model
        if self.n_channels == 0:
            x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn),
                      dtype='float32', name='input')
            y = Lambda(lambda x: K.expand_dims(x, -1), name='lambda')(x)
        else:
            x = Input(
                shape=(self.n_frames_cnn, self.n_freq_cnn, self.n_channels),
                dtype='float32', name='input'
            )
            y = Lambda(lambda x: x, name='lambda')(x)

        # CONV 1
        y = Conv2D(24, self.filter_size_cnn, padding='valid',
                   activation='relu', name='conv1')(y)
        y = MaxPooling2D(pool_size=(2, 2), strides=None,
                         padding='valid', name='maxpool1')(y)
        y = BatchNormalization(name='batchnorm1')(y)

        # CONV 2
        y = Conv2D(48, self.filter_size_cnn, padding='valid',
                   activation='relu', name='conv2')(y)
        y = MaxPooling2D(pool_size=(4, 2), strides=None,
                         padding='valid', name='maxpool2')(y)
        y = BatchNormalization(name='batchnorm2')(y)

        # CONV 3
        y = Conv2D(48, self.filter_size_cnn, padding='valid',
                   activation='relu', name='conv3')(y)
        y = BatchNormalization(name='batchnorm3')(y)

        # Flatten and dense layers
        y = Flatten(name='flatten')(y)
        y = Dropout(0.5, name='dropout1')(y)
        y = Dense(self.n_dense_cnn, activation='relu', kernel_regularizer=l2(
            0.001), bias_regularizer=l2(0.001), name='dense1')(y)
        y = Dropout(0.5, name='dropout2')(y)
        y = Dense(self.n_classes, activation='softmax', kernel_regularizer=l2(
            0.001), bias_regularizer=l2(0.001), name='out')(y)

        # creates keras Model
        self.model = Model(inputs=x, outputs=y)

    def sub_model(self):
        """ Missing docstring here
        """
        # example code on how define a new model based on the original
        new_model = Model(inputs=self.model.input,
                          outputs=self.model.get_layer('dense1').output)
        return new_model

    # def train(...):  # i.e if want to redefine train function


class SB_CNN_SED(KerasModelContainer):
    """ KerasModelContainer for SB_CNN_SED model.

    J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello.
    "Scaper: A Library for Soundscape Synthesis and Augmentation".
    IEEE Workshop on Applications of Signal Processing to
    Audio and Acoustics (WASPAA).
    New Paltz, NY, USA, Oct. 2017

    Notes
    -----
    Code based on Salamon's implementation
    https://github.com/justinsalamon/scaper_waspaa2017

    Parameters
    ----------
    n_classes : int, default=10
        Number of classes (dimmension output).

    n_frames_cnn : int or None, default=64
        Length of the input (number of frames of each sequence).

    n_freq_cnn : int, default=128
        Number of frequency bins. The model's input has shape
        (n_frames, n_freqs).

    filter_size_cnn : tuple, default=(5,5)
        Kernel dimmension for convolutional layers.

    pool_size_cnn : tuple, default=(2,2)
        Pooling dimmension for maxpooling layers.

    large_cnn : bool, default=False
        If large_cnn is true, add other dense layer after penultimate layer.

    n_dense_cnn : int, default=64
        Dimmension of penultimate dense layer.

    n_channels : int, default=0
        Number of input channels.

        0 : mono signals.
            Input shape = (n_frames_cnn, n_freq_cnn)
        1 : mono signals.
            Input shape = (n_frames_cnn, n_freq_cnn, 1)
        2 : stereo signals.
            Input shape = (n_frames_cnn, n_freq_cnn, 2)
        n > 2 : multi-representations.
            Input shape = (n_frames_cnn, n_freq_cnn, n_channels)


    Attributes
    ----------
    model : keras.models.Model
        Keras model.

    Examples
    --------
    >>> from dcase_models.model.models import SB_CNN_SED
    >>> model_container = SB_CNN_SED()
    >>> model_container.model.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 64, 128)           0
    _________________________________________________________________
    lambda_1 (Lambda)            (None, 64, 128, 1)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 60, 124, 64)       1664
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 30, 62, 64)        0
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 30, 62, 64)        256
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 26, 58, 64)        102464
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 29, 64)        0
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 13, 29, 64)        256
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 9, 25, 64)         102464
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 9, 25, 64)         256
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 14400)             0
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 14400)             0
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                921664
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 64)                0
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                650
    =================================================================
    Total params: 1,129,674
    Trainable params: 1,129,290
    Non-trainable params: 384
    _________________________________________________________________

    """

    def __init__(self, model=None, model_path=None, metrics=['sed'],
                 n_classes=10, n_frames_cnn=64,
                 n_freq_cnn=128, filter_size_cnn=(5, 5), pool_size_cnn=(2, 2),
                 large_cnn=False, n_dense_cnn=64,
                 n_filters_cnn=64, n_chanels=0):
        """ Initialization of the SB-CNN-SED model.

        """

        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn
        self.n_freq_cnn = n_freq_cnn
        self.filter_size_cnn = filter_size_cnn
        self.pool_size_cnn = pool_size_cnn
        self.large_cnn = large_cnn
        self.n_dense_cnn = n_dense_cnn
        self.n_filters_cnn = n_filters_cnn
        self.n_chanels = n_chanels

        super().__init__(model=model, model_path=model_path,
                         model_name='SB_CNN_SED', metrics=metrics)

    def build(self):
        """ Missing docstring here
        """
        # Here define the keras model
        if self.large_cnn:
            self.n_filters_cnn = 128
            self.n_dense_cnn = 128

        # INPUT
        x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32')

        y = Lambda(lambda x: K.expand_dims(x, -1))(x)

        # CONV 1
        y = Conv2D(self.n_filters_cnn, self.filter_size_cnn, padding='valid',
                   activation='relu')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn,
                         strides=None, padding='valid')(y)
        y = BatchNormalization()(y)

        # CONV 2
        y = Conv2D(self.n_filters_cnn, self.filter_size_cnn, padding='valid',
                   activation='relu')(y)
        y = MaxPooling2D(pool_size=self.pool_size_cnn,
                         strides=None, padding='valid')(y)
        y = BatchNormalization()(y)

        # CONV 3
        y = Conv2D(self.n_filters_cnn, self.filter_size_cnn, padding='valid',
                   activation='relu')(y)
        # y = MaxPooling2D(pool_size=pool_size_cnn,
        #                  strides=None, padding='valid')(y)
        y = BatchNormalization()(y)

        # Flatten for dense layers
        y = Flatten()(y)
        y = Dropout(0.5)(y)
        y = Dense(self.n_dense_cnn, activation='relu')(y)
        if self.large_cnn:
            y = Dropout(0.5)(y)
            y = Dense(self.n_dense_cnn, activation='relu')(y)
        y = Dropout(0.5)(y)
        y = Dense(self.n_classes, activation='sigmoid')(y)

        self.model = Model(inputs=x, outputs=y)
        super().build()


class A_CRNN(KerasModelContainer):
    """ KerasModelContainer for A_CRNN model.

    S. Adavanne, P. Pertilä, T. Virtanen
    "Sound event detection using spatial features and
    convolutional recurrent neural network"
    International Conference on Acoustics, Speech, and Signal Processing.
    2017. https://arxiv.org/pdf/1706.02291.pdf

    Notes
    -----
    Code based on Adavanne's implementation
    https://github.com/sharathadavanne/sed-crnn

    Parameters
    ----------
    n_classes : int, default=10
        Number of classes (dimmension output).

    n_frames_cnn : int or None, default=64
        Length of the input (number of frames of each sequence).

    n_freq_cnn : int, default=128
        Number of frequency bins. The model's input has shape
        (n_frames, n_freqs).

    cnn_nb_filt : int, default=128
        Number of filters used in convolutional layers.

    cnn_pool_size : tuple, default=(5, 2, 2)
        Pooling dimmension for maxpooling layers.

    rnn_nb : list, default=[32, 32]
        Number of units in each recursive layer.

    fc_nb : list, default=[32]
        Number of units in each dense layer.

    dropout_rate : float, default=0.5
        Dropout rate.

    n_channels : int, default=0
        Number of input channels

        0 : mono signals.
            Input shape = (n_frames_cnn, n_freq_cnn)
        1 : mono signals.
            Input shape = (n_frames_cnn, n_freq_cnn, 1)
        2 : stereo signals.
            Input shape = (n_frames_cnn, n_freq_cnn, 2)
        n > 2 : multi-representations.
            Input shape = (n_frames_cnn, n_freq_cnn, n_channels)

    final_activation : str, default='softmax'
        Activation of the last layer.

    sed : bool, default=False
        If sed is True, the output is frame-level. If False the output is
        time averaged.

    bidirectional : bool, default=False
        If bidirectional is True, the recursive layers are bidirectional.

    Attributes
    ----------
    model : keras.models.Model
        Keras model.

    Examples
    --------
    >>> from dcase_models.model.models import A_CRNN
    >>> model_container = A_CRNN()
    >>> model_container.model.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 64, 128)           0
    _________________________________________________________________
    lambda (Lambda)              (None, 64, 128, 1)        0
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 64, 128, 128)      1280
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 64, 128, 128)      512
    _________________________________________________________________
    activation_4 (Activation)    (None, 64, 128, 128)      0
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 64, 25, 128)       0
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 64, 25, 128)       0
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 64, 25, 128)       147584
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 64, 25, 128)       100
    _________________________________________________________________
    activation_5 (Activation)    (None, 64, 25, 128)       0
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 64, 12, 128)       0
    _________________________________________________________________
    dropout_10 (Dropout)         (None, 64, 12, 128)       0
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 64, 12, 128)       147584
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 64, 12, 128)       48
    _________________________________________________________________
    activation_6 (Activation)    (None, 64, 12, 128)       0
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 64, 6, 128)        0
    _________________________________________________________________
    dropout_11 (Dropout)         (None, 64, 6, 128)        0
    _________________________________________________________________
    reshape_2 (Reshape)          (None, 64, 768)           0
    _________________________________________________________________
    gru_3 (GRU)                  (None, 64, 32)            76896
    _________________________________________________________________
    gru_4 (GRU)                  (None, 64, 32)            6240
    _________________________________________________________________
    time_distributed_6 (TimeDist (None, 64, 32)            1056
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 64, 32)            0
    _________________________________________________________________
    time_distributed_7 (TimeDist (None, 64, 10)            330
    _________________________________________________________________
    mean (Lambda)                (None, 10)                0
    _________________________________________________________________
    strong_out (Activation)      (None, 10)                0
    =================================================================
    Total params: 381,630
    Trainable params: 381,300
    Non-trainable params: 330
    _________________________________________________________________

    """

    def __init__(self, model=None, model_path=None, metrics=['sed'],
                 n_classes=10, n_frames_cnn=64,
                 n_freq_cnn=128, cnn_nb_filt=128,
                 cnn_pool_size=[5, 2, 2], rnn_nb=[32, 32],
                 fc_nb=[32], dropout_rate=0.5, n_channels=0,
                 final_activation='softmax', sed=False,
                 bidirectional=False):
        '''


        '''
        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn
        self.n_freq_cnn = n_freq_cnn
        self.cnn_nb_filt = cnn_nb_filt
        self.cnn_pool_size = cnn_pool_size
        self.rnn_nb = rnn_nb
        self.fc_nb = fc_nb
        self.dropout_rate = dropout_rate
        self.n_channels = n_channels
        self.final_activation = final_activation
        self.sed = sed
        self.bidirectional = bidirectional

        super().__init__(
            model=model, model_path=model_path,
            model_name='A_CRNN', metrics=metrics
        )

    def build(self):
        """ Builds the CRNN Keras model.
        """
        if self.n_channels == 0:
            x = Input(shape=(self.n_frames_cnn, self.n_freq_cnn),
                      dtype='float32', name='input')
            spec_start = Lambda(
                lambda x: K.expand_dims(x, -1), name='lambda')(x)
        else:
            x = Input(
                shape=(self.n_frames_cnn, self.n_freq_cnn, self.n_channels),
                dtype='float32', name='input'
            )
            spec_start = Lambda(lambda x: x, name='lambda')(x)

        spec_x = spec_start
        for i, cnt in enumerate(self.cnn_pool_size):
            spec_x = Conv2D(filters=self.cnn_nb_filt, kernel_size=(
                3, 3), padding='same')(spec_x)
            # spec_x = BatchNormalization(axis=1)(spec_x)
            spec_x = BatchNormalization(axis=2)(spec_x)
            spec_x = Activation('relu')(spec_x)
            spec_x = MaxPooling2D(pool_size=(1, cnt))(spec_x)
            spec_x = Dropout(self.dropout_rate)(spec_x)
        # spec_x = Permute((2, 1, 3))(spec_x)
        spec_x = Reshape((self.n_frames_cnn, -1))(spec_x)

        for r in self.rnn_nb:
            if self.bidirectional:
                spec_x = Bidirectional(
                    GRU(r, activation='tanh', dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate,
                        return_sequences=True),
                    merge_mode='mul')(spec_x)
            else:
                spec_x = GRU(r, activation='tanh', dropout=self.dropout_rate,
                             recurrent_dropout=self.dropout_rate,
                             return_sequences=True)(spec_x)

        for f in self.fc_nb:
            spec_x = TimeDistributed(Dense(f))(spec_x)
            spec_x = Dropout(self.dropout_rate)(spec_x)

        spec_x = TimeDistributed(Dense(self.n_classes))(spec_x)

        if not self.sed:
            spec_x = Lambda(lambda x: K.mean(x, 1), name='mean')(spec_x)
        out = Activation(self.final_activation, name='strong_out')(spec_x)

        # out = Activation('sigmoid', name='strong_out')(spec_x)

        self.model = Model(inputs=x, outputs=out)

        super().build()


class VGGish(KerasModelContainer):
    """ KerasModelContainer for VGGish model

    Jort F. Gemmeke et al.
    Audio Set: An ontology and human-labeled dataset for audio events
    International Conference on Acoustics, Speech, and Signal Processing.
    New Orleans, LA, 2017.

    Notes
    -----
    https://research.google.com/audioset/
    Based on vggish-keras https://pypi.org/project/vggish-keras/

    Parameters
    ----------
    n_frames_cnn : int or None, default=96
        Length of the input (number of frames of each sequence).

    n_freq_cnn : int, default=64
        Number of frequency bins. The model's input has shape
        (n_frames, n_freqs).

    n_classes : int, default=10
        Number of classes (dimmension output).

    n_channels : int, default=0
        Number of input channels

        0 : mono signals.
            Input shape = (n_frames_cnn, n_freq_cnn)
        1 : mono signals.
            Input shape = (n_frames_cnn, n_freq_cnn, 1)
        2 : stereo signals.
            Input shape = (n_frames_cnn, n_freq_cnn, 2)
        n > 2 : multi-representations.
            Input shape = (n_frames_cnn, n_freq_cnn, n_channels)

    embedding_size : int, default=128
        Number of units in the embeddings layer.

    pooling : {'avg', max}, default='avg'
        Use AveragePooling or Maxpooling.

    include_top : bool, default=False
        Include fully-connected layers.

    compress : bool, default=False
        Apply PCA.


    Attributes
    ----------
    model : keras.models.Model
        Keras model.

    Examples
    --------
    >>> from dcase_models.model.models import VGGish
    >>> model_container = VGGish()
    >>> model_container.model.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           (None, 96, 64)            0
    _________________________________________________________________
    lambda (Lambda)              (None, 96, 64, 1)         0
    _________________________________________________________________
    conv1 (Conv2D)               (None, 96, 64, 64)        640
    _________________________________________________________________
    pool1 (MaxPooling2D)         (None, 48, 32, 64)        0
    _________________________________________________________________
    conv2 (Conv2D)               (None, 48, 32, 128)       73856
    _________________________________________________________________
    pool2 (MaxPooling2D)         (None, 24, 16, 128)       0
    _________________________________________________________________
    conv3/conv3_1 (Conv2D)       (None, 24, 16, 256)       295168
    _________________________________________________________________
    conv3/conv3_2 (Conv2D)       (None, 24, 16, 256)       590080
    _________________________________________________________________
    pool3 (MaxPooling2D)         (None, 12, 8, 256)        0
    _________________________________________________________________
    conv4/conv4_1 (Conv2D)       (None, 12, 8, 512)        1180160
    _________________________________________________________________
    conv4/conv4_2 (Conv2D)       (None, 12, 8, 512)        2359808
    _________________________________________________________________
    pool4 (MaxPooling2D)         (None, 6, 4, 512)         0
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 512)               0
    =================================================================
    Total params: 4,499,712
    Trainable params: 4,499,712
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, model=None, model_path=None, metrics=['classification'],
                 n_frames_cnn=96, n_freq_cnn=64, n_classes=10,
                 n_channels=0, embedding_size=128, pooling='avg',
                 include_top=False, compress=False):

        self.n_frames_cnn = n_frames_cnn
        self.n_freq_cnn = n_freq_cnn
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.embedding_size = embedding_size
        self.pooling = pooling
        self.include_top = include_top
        self.compress = compress

        super().__init__(
            model=model, model_path=model_path,
            model_name='VGGish', metrics=metrics
        )

    class Postprocess(Layer):
        """ Keras layer that applies PCA and quantizes the ouput.

        Based on vggish-keras https://pypi.org/project/vggish-keras/
        """
        def __init__(self, output_shape=None, **kw):
            self.emb_shape = output_shape
            super().__init__(**kw)

        def build(self, input_shape):
            input_shape = tuple(int(x) for x in tuple(input_shape)[1:])
            emb_shape = (self.emb_shape,) if self.emb_shape else input_shape

            self.pca_matrix = self.add_weight(
                name='pca_matrix', shape=emb_shape + input_shape,
                initializer='uniform')
            self.pca_means = self.add_weight(
                name='pca_means', shape=input_shape + (1,),
                initializer='uniform')

        def call(self, x):
            # Apply PCA.
            # - Embeddings come in as [batch_size, embedding_size].
            # - Transpose to [embedding_size, batch_size].
            # - Subtract pca_means column vector from each column.
            # - Premultiply by PCA matrix of shape [output_dims, input_dims]
            #   where both are are equal to embedding_size in our case.
            # - Transpose result back to [batch_size, embedding_size].
            x = K.dot(self.pca_matrix, (K.transpose(x) - self.pca_means))
            x = K.transpose(x)

            # Quantize by:
            # - clipping to [min, max] range
            # - convert to 8-bit in range [0.0, 255.0]
            # - cast 8-bit float to uint8
            QUANTIZE_MIN_VAL = -2.0
            QUANTIZE_MAX_VAL = +2.0
            x = clip_by_value(x, QUANTIZE_MIN_VAL, QUANTIZE_MAX_VAL)
            x = ((x - QUANTIZE_MIN_VAL) *
                 (255.0 / (QUANTIZE_MAX_VAL - QUANTIZE_MIN_VAL)))
            return K.cast(x, 'uint8')

    def build(self):
        """ Builds the VGGish Keras model.
        """
        if self.n_channels == 0:
            inputs = Input(shape=(self.n_frames_cnn, self.n_freq_cnn),
                           dtype='float32', name='input')
            x = Lambda(
                lambda x: K.expand_dims(x, -1), name='lambda'
            )(inputs)
        else:
            inputs = Input(
                shape=(self.n_frames_cnn, self.n_freq_cnn, self.n_channels),
                dtype='float32', name='input'
            )
            x = Lambda(lambda x: x, name='lambda')(inputs)

        # setup layer params
        conv = partial(Conv2D, kernel_size=(3, 3), strides=(
            1, 1), activation='relu', padding='same')
        maxpool = partial(MaxPooling2D, pool_size=(2, 2),
                          strides=(2, 2), padding='same')

        # Block 1
        x = conv(64, name='conv1')(x)
        x = maxpool(name='pool1')(x)

        # Block 2
        x = conv(128, name='conv2')(x)
        x = maxpool(name='pool2')(x)

        # Block 3
        x = conv(256, name='conv3/conv3_1')(x)
        x = conv(256, name='conv3/conv3_2')(x)
        x = maxpool(name='pool3')(x)

        # Block 4
        x = conv(512, name='conv4/conv4_1')(x)
        x = conv(512, name='conv4/conv4_2')(x)
        x = maxpool(name='pool4')(x)

        if self.include_top:
            dense = partial(Dense, activation='relu')

            # FC block
            x = Flatten(name='flatten_')(x)
            x = dense(4096, name='fc1/fc1_1')(x)
            x = dense(4096, name='fc1/fc1_2')(x)
            x = dense(self.embedding_size, name='fc2')(x)

            if self.compress:
                x = self.Postprocess()(x)
        else:
            globalpool = (
                GlobalAveragePooling2D() if self.pooling == 'avg' else
                GlobalMaxPooling2D() if self.pooling == 'max' else None)

            if globalpool:
                x = globalpool(x)

        # Create model
        self.model = Model(inputs, x, name='vggish_model')

        super().build()

    def load_pretrained_model_weights(self,
                                      weights_folder='./pretrained_weights'):
        """
        Loads pretrained weights to self.model weights.

        Parameters
        ----------
        weights_folder : str
            Path to load the weights file

        """
        basepath = os.path.dirname(__file__)
        weights_file = self.model_name + '.hdf5'
        weights_path = os.path.join(basepath, weights_folder, weights_file)
        if not os.path.isfile(weights_path):
            self.download_pretrained_weights()
        self.model.load_weights(weights_path, by_name=True)

    def download_pretrained_weights(self,
                                    weights_folder='./pretrained_weights'):
        """
        Download pretrained weights from:
        https://github.com/DTaoo/VGGish
        https://drive.google.com/file/d/1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6/view

        Code based on:
        https://github.com/beasteers/VGGish/blob/master/vggish_keras/download_helpers/download_weights.py

        Parameters
        ----------
        weights_folder : str
            Path to save the weights file

        """
        import requests
        import tqdm

        DRIVE_URL = 'https://drive.google.com/uc?id={id}&export=download'
        DRIVE_CONFIRM_URL = ('https://drive.google.com/uc?id={id}&export'
                             '=download&confirm={confirm}')

        basepath = os.path.dirname(__file__)
        weights_file = self.model_name + '.hdf5'
        weights_path = os.path.join(basepath, weights_folder, weights_file)
        # gdrive_id = '1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6'
        # This file includes PCA weights
        gdrive_id = '1QbMNrhu4RBUO6hIcpLqgeuVye51XyMKM'

        if not os.path.isfile(weights_path):
            print('Downloading weights...')

            sess = requests.Session()
            r = sess.get(DRIVE_URL.format(id=gdrive_id), stream=True)

            # check for google virus message
            confirm = next(
                (v for k, v in r.cookies.get_dict().items()
                 if 'download_warning_' in k), None)

            if confirm:
                # print('Using confirmation code {}...'.format(confirm))
                r = sess.get(
                    DRIVE_CONFIRM_URL.format(id=gdrive_id, confirm=confirm),
                    stream=True)

            # download w/ progress bar

            chunk_size = 1024
            unit = 1024 ** 2
            with open(weights_path, 'wb') as f:
                pbar = tqdm.tqdm(
                    unit='mb', leave=False,
                    total=int(
                        r.headers.get('Content-Length', 0)) / unit or None)

                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        pbar.update(len(chunk) / unit)
                        f.write(chunk)

            print('Done!')


class SMel(KerasModelContainer):
    """ KerasModelContainer for SMel model.

    P. Zinemanas, P. Cancela, M. Rocamora.
    "End–to–end Convolutional Neural Networks for Sound Event Detection
    in Urban Environments"
    Proceedings of the 24th Conference of Open Innovations Association FRUCT,
    3rd IEEE FRUCT International Workshop on Semantic Audio
    and the Internet of Things.
    Moscow, Russia, April 2019.

    Parameters
    ----------
    mel_bands : int, default=128
        Number of mel bands.

    n_seqs : int, default=64
        Time dimmension of the input.

    audio_win : int, default=1024
        Length of the audio window (number of samples of each frame).

    audio_hop : int, default=512
        Length of the hop size (in samples).

    alpha : int, default=1
        Multiply factor before apply log (compression factor).

    scaler : tuple, list or None
        If scaler is not None, this is used before output.

    amin : float, default=1e-10 (-100 dB)
        Minimum value for db calculation.

    Attributes
    ----------
    model : keras.models.Model
        Keras model.

    Examples
    --------
    >>> from dcase_models.model.models import SMel
    >>> model_container = SMel()
    >>> model_container.model.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 64, 1024)          0
    _________________________________________________________________
    lambda (Lambda)              (None, 64, 1024, 1)       0
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 64, 64, 128)       131200
    _________________________________________________________________
    lambda_1 (Lambda)            (None, 64, 64, 128)       0
    _________________________________________________________________
    lambda_2 (Lambda)            (None, 64, 128)           0
    _________________________________________________________________
    lambda_3 (Lambda)            (None, 64, 128)           0
    =================================================================
    Total params: 131,200
    Trainable params: 131,200
    Non-trainable params: 0
    _________________________________________________________________

    """

    def __init__(self, model=None, model_path=None,
                 metrics=['mean_squared_error'],
                 mel_bands=128, n_seqs=64,
                 audio_win=1024, audio_hop=512,
                 alpha=1, scaler=None, amin=1e-10):
        self.mel_bands = mel_bands
        self.n_seqs = n_seqs
        self.audio_win = audio_win
        self.audio_hop = audio_hop
        self.alpha = alpha
        self.scaler = scaler
        self.amin = amin

        super().__init__(model=model, model_path=model_path,
                         model_name='SMel', metrics=metrics)

    def build(self):
        """ Builds the SMel Keras model.
        """
        x = Input(shape=(self.n_seqs, self.audio_win), dtype='float32')

        y = Lambda(lambda x: K.expand_dims(x, -1), name='lambda')(x)

        y = TimeDistributed(
            Conv1D(
                self.mel_bands, 1024, strides=16, padding='same', use_bias=True
            ))(y)

        y = Lambda(lambda x: x*x)(y)

        y = Lambda(lambda x: self.audio_win*K.mean(x, axis=2))(y)

        y = Lambda(
            lambda x: 10*K.log(K.maximum(self.amin, x*self.alpha))/K.log(10.)
        )(y)

        if self.scaler is not None:
            y = Lambda(
                lambda x: 2*((x-self.scaler[0]) /
                             (self.scaler[1]-self.scaler[0])-0.5)
            )(y)

        self.model = Model(inputs=x, outputs=y)

        super().build()


class MST(KerasModelContainer):
    """ KerasModelContainer for MST model.

    T. M. S. Tax, J. L. D. Antich, H. Purwins, and L. Maaløe.
    “Utilizing domain knowledge in end-to-end audio processing”
    31st Conference on Neural Information Processing Systems (NIPS).
    Long Beach, CA, USA, 2017.

    Parameters
    ----------
    mel_bands : int, default=128
        Number of mel bands.

    sequence_samples : int, default=22050
        Number of samples of each input.

    audio_win : int, default=1024
        Length of the audio window (number of samples of each frame).

    audio_hop : int, default=512
        Length of the hop size (in samples).


    Attributes
    ----------
    model : keras.models.Model
        Keras model.

    Examples
    --------
    >>> from dcase_models.model.models import SMel
    >>> model_container = SMel()
    >>> model_container.model.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_2 (InputLayer)         (None, 22050)             0
    _________________________________________________________________
    lambda (Lambda)              (None, 22050, 1)          0
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 44, 512)           524800
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 44, 512)           2048
    _________________________________________________________________
    activation_1 (Activation)    (None, 44, 512)           0
    _________________________________________________________________
    conv1d_3 (Conv1D)            (None, 44, 256)           393472
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 44, 256)           1024
    _________________________________________________________________
    activation_2 (Activation)    (None, 44, 256)           0
    _________________________________________________________________
    conv1d_4 (Conv1D)            (None, 44, 128)           98432
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 44, 128)           512
    _________________________________________________________________
    activation_3 (Activation)    (None, 44, 128)           0
    =================================================================
    Total params: 1,020,288
    Trainable params: 1,018,496
    Non-trainable params: 1,792
    _________________________________________________________________
    """

    def __init__(self, model=None, model_path=None,
                 metrics=['mean_squared_error'],
                 mel_bands=128, sequence_samples=22050,
                 audio_win=1024, audio_hop=512):
        self.mel_bands = mel_bands
        self.sequence_samples = sequence_samples
        self.audio_win = audio_win
        self.audio_hop = audio_hop

        super().__init__(model=model, model_path=model_path,
                         model_name='MST', metrics=metrics)

    def build(self):
        """ Builds the MST Keras model.
        """
        x = Input(shape=(self.sequence_samples, ), dtype='float32')

        y = Lambda(lambda x: K.expand_dims(x, -1), name='lambda')(x)

        y = Conv1D(512, self.audio_win,
                   strides=self.audio_hop, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(256, 3, strides=1, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(self.mel_bands, 3, strides=1, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('tanh')(y)

        self.model = Model(inputs=x, outputs=y)

        super().build()


class ConcatenatedModel(KerasModelContainer):
    """ KerasModelContainer for concatenating models.

    """

    def __init__(self, model_list, model_path=None,
                 model_name='ConcatenatedModel', metrics=['sed'],
                 use_batch_norm=False):
        """ Initialization of ConcatenatedModel.

        """
        self.model_list = model_list
        self.use_batch_norm = use_batch_norm

        super().__init__(model=None, model_path=model_path,
                         model_name=model_name, metrics=metrics)

    def build(self):
        """ Missing docstring here
        """
        input_shape = self.model_list[0].model.input_shape
        print(input_shape)
        x = Input(shape=input_shape[1:], dtype='float32')
        for j in range(len(self.model_list)):
            if j == 0:
                y = x
            print(y.shape)
            y = self.model_list[j].model(y)
            print(y.shape)
            if self.use_batch_norm and (j < len(self.model_list) - 1):
                y = BatchNormalization()(y)

        self.model = Model(inputs=x, outputs=y)
        super().build()


def get_available_models():
    """ Missing docstring here
    """
    available_models = {m[0]: m[1] for m in inspect.getmembers(
        sys.modules[__name__], inspect.isclass) if m[1].__module__ == __name__}

    return available_models
