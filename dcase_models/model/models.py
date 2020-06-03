from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

from .container import DCASEModelContainer

class SB_CNN(DCASEModelContainer):
    """
    Inherit class of DCASEModelContainer with specific attributs and methods for
    SB_CNN model.

    """

    def __init__(self, model=None, folder=None, metrics=['accuracy'], n_classes=10, n_frames_cnn=64, 
                n_freq_cnn=128, filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                large_cnn=False, n_dense_cnn=64, n_channels=0): 
        """
        Function that init the SB-CNN [1] model.
        
        [1] ​Deep Convolutional Neural Networks and Data Augmentation For Environmental Sound Classification
​            J. Salamon and J. P. Bello
            IEEE Signal Processing Letters, 24(3), pages 279 - 283, 2017.
        
        ----------
        model : keras Model
            
        n_freq_cnn : int
            number of frecuency bins of the input
        n_frames_cnn : int
            number of time steps (hops) of the input
        n_filters_cnn : int
            number of filter in each conv layer
        filter_size_cnn : tuple of int
            kernel size of each conv filter
        pool_size_cnn : tuple of int
            kernel size of the pool operations
        n_classes : int
            number of classes for the classification taks 
            (size of the last layer)
        large_cnn : bool
            If true, the model has one more dense layer 
        n_dense_cnn : int
            Size of middle dense layers

        Notes
        -----
        Code based on Salamon's implementation 
        https://github.com/justinsalamon/scaper_waspaa2017
        
        """ 
        if folder is None:
            ### Here define the keras model
            # INPUT
            if n_channels == 0:
                x = Input(shape=(n_frames_cnn,n_freq_cnn), dtype='float32', name='input')
                y = Lambda(lambda x: K.expand_dims(x,-1), name='lambda')(x) 
            else:
                x = Input(shape=(n_frames_cnn,n_freq_cnn, n_chanels), dtype='float32', name='input')
                y = Lambda(lambda x: x, name='lambda')(x) 

            # CONV 1
            y = Conv2D(24, filter_size_cnn, padding='valid', activation='relu', name='conv1')(y)
            y = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', name='maxpool1')(y)
            y = BatchNormalization(name='batchnorm1')(y)

            # CONV 2
            y = Conv2D(48, filter_size_cnn, padding='valid', activation='relu', name='conv2')(y)
            y = MaxPooling2D(pool_size=(4,2), strides=None, padding='valid', name='maxpool2')(y)
            y = BatchNormalization(name='batchnorm2')(y)

            # CONV 3
            y = Conv2D(48, filter_size_cnn, padding='valid', activation='relu', name='conv3')(y)
            y = BatchNormalization(name='batchnorm3')(y)

            # Flatten and dense layers
            y = Flatten(name='flatten')(y)
            y = Dropout(0.5,name='dropout1')(y)
            y = Dense(n_dense_cnn, activation='relu',kernel_regularizer=l2(0.001),bias_regularizer=l2(0.001),name='dense1')(y)
            y = Dropout(0.5,name='dropout2')(y)
            y = Dense(n_classes, activation='softmax',kernel_regularizer=l2(0.001),bias_regularizer=l2(0.001),name='out')(y)

            # creates keras Model
            model = Model(inputs=x, outputs=y)

        super().__init__(model=model, folder=folder, model_name='SB_CNN', metrics=metrics)

    def sub_model():
        # example code on how define a new model based on the original
        new_model = Model(inputs=self.model.input, outputs=self.model.get_layer('dense1').output)
        return new_model

    # def train(...):  # i.e if want to redefine train function

class SB_CNN_SED(DCASEModelContainer):
    """
    Inherit class of DCASEModelContainer with specific attributs and methods for
    SB_CNN_SED model.

    """

    def __init__(self, model=None, folder=None, metrics=['accuracy'], n_classes=10, n_frames_cnn=64, 
                n_freq_cnn=128, filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                large_cnn=False, n_dense_cnn=64, n_filters_cnn=64, n_chanels=0): 
        """
        Function that init the SB-CNN-SED [2] model.
        
        [2] Scaper: A Library for Soundscape Synthesis and Augmentation
            J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello.
            In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2017

        ----------
        model : keras Model
            
        n_freq_cnn : int
            number of frecuency bins of the input
        n_frames_cnn : int
            number of time steps (hops) of the input
        n_filters_cnn : int
            number of filter in each conv layer
        filter_size_cnn : tuple of int
            kernel size of each conv filter
        pool_size_cnn : tuple of int
            kernel size of the pool operations
        n_classes : int
            number of classes for the classification taks 
            (size of the last layer)
        large_cnn : bool
            If true, the model has one more dense layer 
        n_dense_cnn : int
            Size of middle dense layers

        Notes
        -----
        Code based on Salamon's implementation 
        https://github.com/justinsalamon/scaper_waspaa2017
        
        """ 
        if folder is None:
            ### Here define the keras model
            if large_cnn:
                n_filters_cnn = 128
                n_dense_cnn = 128

            # INPUT
            x = Input(shape=(n_frames_cnn,n_freq_cnn), dtype='float32')
                        
            y = Lambda(lambda x: K.expand_dims(x,-1))(y) 
            # CONV 1
            y = Conv2D(n_filters_cnn, filter_size_cnn, padding='valid',
                    activation='relu')(y)
            y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
            y = BatchNormalization()(y)

            # CONV 2
            y = Conv2D(n_filters_cnn, filter_size_cnn, padding='valid',
                    activation='relu')(y)
            y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
            y = BatchNormalization()(y)

            # CONV 3
            y = Conv2D(n_filters_cnn, filter_size_cnn, padding='valid',
                    activation='relu')(y)
            # y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
            y = BatchNormalization()(y)

            # Flatten for dense layers
            y = Flatten()(y)
            y = Dropout(0.5)(y)
            y = Dense(n_dense_cnn, activation='relu')(y)
            if large_cnn:
                y = Dropout(0.5)(y)
                y = Dense(n_dense_cnn, activation='relu')(y)
            y = Dropout(0.5)(y)
            y = Dense(n_classes, activation='sigmoid')(y)

            model = Model(inputs=x, outputs=y)
            model.summary()
            
        super().__init__(model=model, folder=folder, model_name='SB_CNN_SED', metrics=metrics)

from keras.layers import GRU, Bidirectional, TimeDistributed, Activation, Permute, Reshape
class A_CRNN(DCASEModelContainer):

    def __init__(self, model=None, folder=None, metrics=['accuracy'], n_classes=10, n_frames_cnn=64, 
                n_freq_cnn=128, cnn_nb_filt=128, cnn_pool_size=[5, 2, 2], rnn_nb = [32, 32],
                fc_nb = [32], dropout_rate = 0.5, n_channels=0, final_activation='softmax', sed=False, bidirectional=False): 

        '''
        Sound event detection using spatial features and convolutional recurrent neural network
        S Adavanne, P Pertilä, T Virtanen
        ICASSP 2017
        
        # based on https://github.com/sharathadavanne/sed-crnn
        # ref https://arxiv.org/pdf/1706.02291.pdf

        '''
        if folder is None:
            if n_channels == 0:
                x = Input(shape=(n_frames_cnn,n_freq_cnn), dtype='float32', name='input')
                spec_start = Lambda(lambda x: K.expand_dims(x,-1), name='lambda')(x) 
            else:
                x = Input(shape=(n_frames_cnn,n_freq_cnn, n_chanels), dtype='float32', name='input')
                spec_start = Lambda(lambda x: x, name='lambda')(x) 

            spec_x = spec_start
            for i, cnt in enumerate(cnn_pool_size):
                spec_x = Conv2D(filters=cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x)
                print(i, spec_x.shape)
                #spec_x = BatchNormalization(axis=1)(spec_x)
                spec_x = BatchNormalization(axis=2)(spec_x)
                spec_x = Activation('relu')(spec_x)
                spec_x = MaxPooling2D(pool_size=(1, cnt))(spec_x)
                spec_x = Dropout(dropout_rate)(spec_x)
            #spec_x = Permute((2, 1, 3))(spec_x)
            spec_x = Reshape((n_frames_cnn, -1))(spec_x)

            for r in rnn_nb:
                if bidirectional:
                    spec_x = Bidirectional(
                        GRU(r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
                        merge_mode='mul')(spec_x)
                else:
                    spec_x = GRU(r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True)(spec_x)                        

            for f in fc_nb:
                spec_x = TimeDistributed(Dense(f))(spec_x)
                spec_x = Dropout(dropout_rate)(spec_x)

            spec_x = TimeDistributed(Dense(n_classes))(spec_x)

            if not sed:
                spec_x = Lambda(lambda x: K.mean(x,1), name='mean')(spec_x)
            out = Activation(final_activation, name='strong_out')(spec_x)
             
            #out = Activation('sigmoid', name='strong_out')(spec_x)

            model = Model(inputs=x, outputs=out)

        super().__init__(model=model, folder=folder, model_name='A_CRNN', metrics=metrics)

from functools import partial
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

class VGGish(DCASEModelContainer):

    '''

    Audio Set: An ontology and human-labeled dataset for audio events
    Jort F. Gemmeke Daniel P. W. Ellis Dylan Freedman Aren Jansen Wade Lawrence R. Channing Moore Manoj Plakal Marvin Ritter
    Proc. IEEE ICASSP 2017, New Orleans, LA

    https://research.google.com/audioset/
    based on vggish-keras https://pypi.org/project/vggish-keras/
    '''
    def __init__(self, model=None, folder=None, metrics=['accuracy'], n_frames_cnn=96, 
                n_freq_cnn=64, n_classes=10, n_channels=0, embedding_size=128, pooling='avg', include_top=False, compress=False):

        if folder is None:
            if n_channels == 0:
                inputs = Input(shape=(n_frames_cnn,n_freq_cnn), dtype='float32', name='input')
                x = Lambda(lambda x: K.expand_dims(x,-1), name='lambda')(inputs) 
            else:
                inputs = Input(shape=(n_frames_cnn,n_freq_cnn, n_chanels), dtype='float32', name='input')
                x = Lambda(lambda x: x, name='lambda')(inputs) 

            # setup layer params
            conv = partial(Conv2D, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')
            maxpool = partial(MaxPooling2D, pool_size=(2, 2), strides=(2, 2), padding='same')

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

            if include_top:
                dense = partial(Dense, activation='relu')

                # FC block
                x = Flatten(name='flatten_')(x)
                x = dense(4096, name='fc1/fc1_1')(x)
                x = dense(4096, name='fc1/fc1_2')(x)
                x = dense(embedding_size, name='fc2')(x)

                if compress:
                    x = Postprocess()(x)
            else:
                globalpool = (
                    GlobalAveragePooling2D() if pooling == 'avg' else
                    GlobalMaxPooling2D() if pooling == 'max' else None)

                if globalpool:
                    x = globalpool(x)

            # Create model
            model = Model(inputs, x, name='vggish_model')


        super().__init__(model=model, folder=folder, model_name='VGGish', metrics=metrics)

from autopool import AutoPool1D
class DCASE2020Task5Baseline(DCASEModelContainer):
    '''

    Baseline of Urban Sound Tagging with Spatiotemporal Context
    DCASE 2020 Challenge - Task 5

    Mark Cartwright, Ana Elisa Mendez Mendez, Jason Cramer, Vincent Lostanlen, Graham Dove, 
    Ho-Hsiang Wu, Justin Salamon, Oded Nov, and Juan Bello. 
    SONYC urban sound tagging (SONYC-UST): a multilabel dataset from an urban acoustic sensor network. 
    In Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 35–39. October 2019

    
    based on https://github.com/sonyc-project/dcase2020task5-uststc-baseline/blob/master/src/classify.py
    '''
    def __init__(self, model=None, folder=None, metrics=['microAUPRC', 'macroAUPRC'], n_frames_cnn=96, 
                n_freq_cnn=64, n_classes=10, hidden_layer_size=128, num_hidden_layers=1, l2_reg=1e-5):

        if folder is None:

            # input
            inputs = Input(shape=(n_frames_cnn,n_freq_cnn), dtype='float32', name='input')

            # Hidden layers
            for idx in range(num_hidden_layers):
                y = TimeDistributed(Dense(hidden_layer_size, activation='relu',
                                        kernel_regularizer=regularizers.l2(l2_reg)),
                                    name='dense_{}'.format(idx+1))(y)

            # Output layer
            y = TimeDistributed(Dense(num_classes, activation='sigmoid',
                                    kernel_regularizer=regularizers.l2(l2_reg)),
                                name='output_t',
                                input_shape=(num_frames, repr_size))(y)

            # Apply autopool over time dimension
            y = AutoPool1D(axis=1, name='output')(y)

            # Create model
            model = Model(inputs=inp, outputs=y, name='model')

        super().__init__(model=model, folder=folder, model_name='DCASE2020Task5Baseline', metrics=metrics)