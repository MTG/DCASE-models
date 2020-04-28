from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from .container import DCASEModelContainer

class SB_CNN(DCASEModelContainer):
    """
    Child class of ModelContainer with specific attributs and methods for
    SB_CNN model.

    Attributes
    ----------
    prototypes : Prototypes
        Instance that includes prototypes information for visualization
        and debugging.
    data_instances : DataInstances
        Instance that includes data information for visualization
        and debugging.

    Methods
    -------
    get_prototypes(X_train, convert_audio_params=None, projection2D=None)
        Extract prototypes from model (embeddings, mel-spectrograms and audio
        if convert_audio_params is not None). Init self.prototypes instance. 

    get_data_instances(X_feat, X_train, Y_train, Files_names_train)
        Init self.data_instances object. Load data instances.
        
    debug_prototypes(self, X_train, force_get_prototypes=False)
        Function to debug the model by eliminating similar prototypes
    """

    def __init__(self, model=None, folder=None, metrics=['accuracy'], n_classes=10, n_frames_cnn=64, 
                n_freq_cnn=128, filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                large_cnn=False, n_dense_cnn=64, n_chanels=0): 
        """
        Function that init the SB-CNN [1] model.
        
        [1] Deep Convolutional Neural Networks and Data Augmentation for Environmental SoundClassification
        
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
            model = self.create_model(n_classes=n_classes, n_frames_cnn=n_frames_cnn, 
                     n_freq_cnn=n_freq_cnn, filter_size_cnn=filter_size_cnn, pool_size_cnn=pool_size_cnn,
                     large_cnn=large_cnn, n_dense_cnn=n_dense_cnn, n_chanels=n_chanels)

        super().__init__(model=model, folder=folder, model_name='SB_CNN', metrics=metrics)

    def create_model(self, n_classes=10, n_frames_cnn=64, 
                     n_freq_cnn=128, filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                     large_cnn=False, n_dense_cnn=64, n_chanels=0):
            ### Here define the keras model
            # INPUT
            if n_chanels == 0:
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
            y = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', name='maxpool2')(y)
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
            return model
    def sub_model():
        # example code on how define a new model based on the original
        new_model = Model(inputs=self.model.input, outputs=self.model.get_layer('dense1').output)
        return new_model


    # def train(...):  # i.e if want to redefine train function