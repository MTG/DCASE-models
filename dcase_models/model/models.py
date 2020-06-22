from functools import partial
import inspect
import sys

from keras.layers import GRU, Bidirectional
from keras.layers import TimeDistributed, Activation, Reshape
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
from autopool import AutoPool1D

from .container import KerasModelContainer


class MLP(KerasModelContainer):
    """ KerasModelContainer for a generic MLP model.

    """

    def __init__(self, model=None, model_path=None,
                 metrics=['accuracy'], n_classes=10,
                 n_frames_cnn=64, n_freq_cnn=12,
                 hidden_layers_size=[128, 64],
                 dropout_rates=[0.5, 0.5], hidden_activation='relu',
                 l2_reg=1e-5, final_activation='softmax',
                 temporal_integration='mean', **kwargs):

        # self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn
        self.n_freq_cnn = n_freq_cnn
        self.hidden_layers_size = hidden_layers_size
        self.dropout_rates = dropout_rates
        self.l2_reg = l2_reg
        self.temporal_integration = temporal_integration
        self.use_time_distributed = n_frames_cnn is not None
        self.hidden_activation = hidden_activation
        self.final_activation = final_activation
        self.kwargs = kwargs

        super().__init__(model=model, model_path=model_path,
                         model_name='MLP', metrics=metrics)

    def build(self):
        # input
        if self.use_time_distributed:
            input_shape = (self.n_frames_cnn, self.n_freq_cnn)
        else:
            input_shape = (self.n_freq_cnn)

        inputs = Input(shape=input_shape, dtype='float32', name='input')

        # Hidden layers
        for idx in range(len(self.hidden_layers_size)):
            if idx == 0:
                y = inputs
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
        if self.temporal_integration == 'mean':
            y = Lambda(lambda x: K.mean(x, 1), name='temporal_integration')(y)
        elif self.temporal_integration == 'sum':
            y = Lambda(lambda x: K.sum(x, 1), name='temporal_integration')(y)
        elif self.temporal_integration == 'autopool':
            y = AutoPool1D(axis=1, name='output')(y)

        # Create model
        self.model = Model(inputs=inputs, outputs=y, name='model')

        super().build()


class SB_CNN(KerasModelContainer):
    """ KerasModelContainer for SB_CNN model.

    J. Salamon and J. P. Bello.
    "Deep Convolutional Neural Networks and Data Augmentation
    For Environmental Sound Classification".
​    IEEE Signal Processing Letters, 24(3), pages 279 - 283.
    2017.

    """

    def __init__(self, model=None, model_path=None, metrics=['accuracy'],
                 n_classes=10, n_frames_cnn=64,
                 n_freq_cnn=128, filter_size_cnn=(5, 5), pool_size_cnn=(2, 2),
                 large_cnn=False, n_dense_cnn=64, n_channels=0):
        """
        Initialization of the SB-CNN [1] model.

        ----------
        model : keras Model

        n_freq_cnn : int
            number of frequency bins of the input
        n_frames_cnn : int
            number of time steps (hops) of the input
        n_filters_cnn : int
            number of filters in each conv layer
        filter_size_cnn : tuple of int
            kernel size of each convolutional filter
        pool_size_cnn : tuple of int
            kernel size of the pool operations
        n_classes : int
            number of classes for the classification task
            (size of the last layer)
        large_cnn : bool
            if true, the model has one more dense layer
        n_dense_cnn : int
            size of middle dense layers

        Notes
        -----
        Code based on Salamon's implementation
        https://github.com/justinsalamon/scaper_waspaa2017

        """

        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn
        self.n_freq_cnn = n_freq_cnn
        self.filter_size_cnn = filter_size_cnn
        self.pool_size_cnn = pool_size_cnn
        self.large_cnn = large_cnn
        self.n_dense_cnn = n_dense_cnn
        self.n_channels = n_channels

        super().__init__(
            model=model, model_path=model_path,
            model_name='SB_CNN', metrics=metrics
        )

    def build(self):
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

    """

    def __init__(self, model=None, model_path=None, metrics=['accuracy'],
                 n_classes=10, n_frames_cnn=64,
                 n_freq_cnn=128, filter_size_cnn=(5, 5), pool_size_cnn=(2, 2),
                 large_cnn=False, n_dense_cnn=64,
                 n_filters_cnn=64, n_chanels=0):
        """
        Initialization of the SB-CNN-SED model.

        ----------
        model : keras Model

        n_freq_cnn : int
            number of frequency bins of the input
        n_frames_cnn : int
            number of time steps (hops) of the input
        n_filters_cnn : int
            number of filters in each conv layer
        filter_size_cnn : tuple of int
            kernel size of each convolutional filter
        pool_size_cnn : tuple of int
            kernel size of the pool operations
        n_classes : int
            number of classes for the classification task
            (size of the last layer)
        large_cnn : bool
            if true, the model has one more dense layer
        n_dense_cnn : int
            size of middle dense layers

        Notes
        -----
        Code based on Salamon's implementation
        https://github.com/justinsalamon/scaper_waspaa2017

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
    2017.

    """

    def __init__(self, model=None, model_path=None, metrics=['accuracy'],
                 n_classes=10, n_frames_cnn=64,
                 n_freq_cnn=128, cnn_nb_filt=128,
                 cnn_pool_size=[5, 2, 2], rnn_nb=[32, 32],
                 fc_nb=[32], dropout_rate=0.5, n_channels=0,
                 final_activation='softmax', sed=False,
                 bidirectional=False):
        '''
        # based on https://github.com/sharathadavanne/sed-crnn
        # ref https://arxiv.org/pdf/1706.02291.pdf

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
            print(i, spec_x.shape)
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

    https://research.google.com/audioset/
    based on vggish-keras https://pypi.org/project/vggish-keras/
    """

    def __init__(self, model=None, model_path=None, metrics=['accuracy'],
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

    def build(self):
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

            # if compress:
            #    x = Postprocess()(x)
        else:
            globalpool = (
                GlobalAveragePooling2D() if self.pooling == 'avg' else
                GlobalMaxPooling2D() if self.pooling == 'max' else None)

            if globalpool:
                x = globalpool(x)

        # Create model
        self.model = Model(inputs, x, name='vggish_model')

        super().build()


class DCASE2020Task5Baseline(KerasModelContainer):
    """ Baseline of Urban Sound Tagging with Spatiotemporal Context
    DCASE 2020 Challenge - Task 5

    Mark Cartwright et al.
    SONYC urban sound tagging (SONYC-UST): a multilabel dataset
    from an urban acoustic sensor network.
    In Proceedings of the Workshop on Detection and Classification of
    Acoustic Scenes and Events (DCASE), 35–39. October 2019

    based on https://github.com/sonyc-project/
             dcase2020task5-uststc-baseline/blob/master/src/classify.py
    """

    def __init__(self, model=None, model_path=None,
                 metrics=['microAUPRC', 'macroAUPRC'], n_frames_cnn=96,
                 n_freq_cnn=64, n_classes=10, hidden_layer_size=128,
                 num_hidden_layers=1, l2_reg=1e-5):

        self.n_frames_cnn = n_frames_cnn
        self.n_freq_cnn = n_freq_cnn
        self.n_classes = n_classes
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.l2_reg = l2_reg

        super().__init__(model=model, model_path=model_path,
                         model_name='DCASE2020Task5Baseline', metrics=metrics)

    def build(self):
        # input
        inputs = Input(shape=(self.n_frames_cnn, self.n_freq_cnn),
                       dtype='float32', name='input')

        # Hidden layers
        for idx in range(self.num_hidden_layers):
            if idx == 0:
                y = inputs
            y = TimeDistributed(
                Dense(self.hidden_layer_size, activation='relu',
                      kernel_regularizer=l2(self.l2_reg)),
                name='dense_{}'.format(idx+1)
                )(y)

        # Output layer
        y = TimeDistributed(Dense(self.n_classes, activation='sigmoid',
                                  kernel_regularizer=l2(self.l2_reg)),
                            name='output_t')(y)

        # Apply autopool over time dimension
        y = AutoPool1D(axis=1, name='output')(y)

        # Create model
        self.model = Model(inputs=inputs, outputs=y, name='model')

        super().build()


def get_available_models():
    available_models = {m[0]: m[1] for m in inspect.getmembers(
        sys.modules[__name__], inspect.isclass) if m[1].__module__ == __name__}

    return available_models
