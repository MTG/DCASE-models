import numpy as np
import os
import json

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import model_from_json
import keras.backend as K

from ..utils.files import save_json
from ..utils.metrics import accuracy
from ..utils.callbacks import MetricsCallback

class ModelContainer():
    """
    A class to contain a keras model, the methods to train, evaluate,
    save and load the model. Child of this class can be created for
    especific models (i.e see APNet class)

    Attributes
    ----------
    model : keras model
        APNet keras model (see apnet.model.apnet())
    folder : str
        if model is None, this path is used to load the model

    Methods
    -------
    train(X_train, Y_train, X_val, Y_val, weights_path= './',  log_path= './',
          loss_weights = [10,5,5], optimizer = 'adam',
          learning_rate = 0.001, batch_size = 256, epochs=100, fit_verbose = 1)
        Train the keras model using the data and paramaters of arguments
    
    evaluate(X_test, Y_test, scaler=None)
        Evaluate the keras model using X_test and Y_test
    
    load_model_from_json(folder):
        Load model from model.json file in the path given by argument

    save_model_json(folder)
        Save a model.json file in the path given by argument

    save_model_weights(weights_folder)
        Save model weights in the path given by argument

    """


    def __init__(self, model=None, folder=None, model_name="APNet"):    
        """
        Parameters
        ----------
        model : keras model
            APNet keras model (see apnet.model.apnet())
        folder : str
            if model is None, this path is used to load the model.
            See load_model_from_json()
        """
        if model is not None:
            self.model = model
        elif folder is not None:
            self.load_model_from_json(folder)
        else:
            raise AttributeError("model or folder are both None")
        self.model_name = model_name

    def train(self, X_train, Y_train, X_val, Y_val, weights_path= './',  log_path= './',
               optimizer = 'Adam',learning_rate = 0.001, early_stopping=100, **kwargs_keras_fit):
        """
        Train the keras model using the data and paramaters of arguments.
        This function runs a function named "train_{model_name}" (i.e train_APNet).

        Parameters
        ----------
        X_train : ndarray
            3D array with mel-spectrograms of train set.
            Shape = (N_instances, N_hops, N_mel_bands)
        Y_train : ndarray
            2D array with the annotations of train set (one hot encoding).
            Shape (N_instances, N_classes)
        X_val : ndarray
            3D array with mel-spectrograms of validation set.
            Shape = (N_instances, N_hops, N_mel_bands)
        Y_val : ndarray
            2D array with the annotations of validation set (one hot encoding).
            Shape (N_instances, N_classes)            
        weights_path : str
            Path where to save the best weights of the model in the training process
        weights_path : str
            Path where to save log of the training process
        loss_weights : list
            List of weights for each loss function ('categorical_crossentropy',
            'mean_squared_error', 'prototype_loss')
        optimizer : str
            Optimizer used to train the model  
        learning_rate : float
            Learning rate used to train the model
        batch_size : int
            Batch size used in the training process
        epochs : int
            Number of epochs of training
        fit_verbose : int
            Verbose of fit method of keras model

        """
        import keras.optimizers as optimizers
        optimizer_function = getattr(optimizers, optimizer)
        opt = optimizer_function(lr=learning_rate)

        self.model.compile(loss='categorical_crossentropy', optimizer=opt)

        file_weights = os.path.join(weights_path, 'best_weights.hdf5') 
        file_log = os.path.join(weights_path, 'training.log') 
        metrics_callback = MetricsCallback(X_val, Y_val, file_weights=file_weights, early_stopping=early_stopping)
        log = CSVLogger(file_log)
        history = self.model.fit(x = X_train, y = Y_train, shuffle = True,
                                 callbacks = [metrics_callback,log], **kwargs_keras_fit)              

    def evaluate(self, X_test, Y_test, scaler = None):
        """
        Evaluate the keras model using X_test and Y_test

        Parameters
        ----------
        X_test : ndarray
            3D array with mel-spectrograms of test set.
            Shape = (N_instances, N_hops, N_mel_bands)
        Y_test : ndarray
            2D array with the annotations of test set (one hot encoding).
            Shape (N_instances, N_classes)
        scaler : Scaler, optional
            Scaler objet to be applied if is not None.

        Returns
        -------
        float
            accuracy of evaluation
        list
            list of annotations (ground_truth)
        list
            list of model predictions 
        
        """
        if scaler is not None:
            X_test = scaler.transform(X_test)
        return accuracy(self.model, X_test, Y_test)

    def load_model_from_json(self, folder, custom_objects=None):
        """
        Load model from model.json file in the path given by argument.
        The model is load in self.model attribute

        Parameters
        ----------
        folder : str
            Path to the folder that contains model.json file
        """
        weights_file = os.path.join(folder, 'best_weights.hdf5')
        json_file = os.path.join(folder, 'model.json')
        
        with open(json_file) as json_f:
            data = json.load(json_f)
        self.model = model_from_json(data,custom_objects=custom_objects)#{'Prototype_distances_separate': Prototype_distances_separate,
                                                         # 'Prototype_distances': Prototype_distances, 'Mean': Mean, 
                                                         # 'Window': Window, 'DecoderFilterIntegration': DecoderFilterIntegration, 'conv_kernel_reg':conv_kernel_reg}
        self.model.load_weights(weights_file)    

    def save_model_json(self,folder):
        """
        Save model to model.json file in the path given by argument.

        Parameters
        ----------
        folder : str
            Path to the folder to save model.json file
        """
        json_string = self.model.to_json()
        json_file = 'model.json'
        json_path = os.path.join(folder, json_file)
        save_json(json_path, json_string)

    def save_model_weights(self, weights_folder):
        """
        Save self.model weights in weights_folder/best_weights.hdf5.

        Parameters
        ----------
        weights_folder : str
            Path to save the weights file
        """
        weights_file = 'best_weights.hdf5'
        weights_path = os.path.join(weights_folder, weights_file)  
        self.model.save_weights(weights_path)

    def load_model_weights(self, weights_folder):
        """
        Save self.model weights in weights_folder/best_weights.hdf5.

        Parameters
        ----------
        weights_folder : str
            Path to save the weights file
        """
        weights_file = 'best_weights.hdf5'
        weights_path = os.path.join(weights_folder, weights_file)  
        self.model.load_weights(weights_path)        

    def get_numer_of_parameters(self):
        trainable_count = int(np.sum([K.count_params(p) for p in set(models.trainable_weights)]))
        return trainable_count


class SB_CNN(ModelContainer):
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

    def __init__(self, model=None, folder=None, n_classes=10, n_frames_cnn=64, 
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
        from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
        from keras.layers.normalization import BatchNormalization
        from keras.models import Model
        from keras.regularizers import l2

        if folder is None:
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
            y = MaxPooling2D(pool_size=(4,2), strides=None, padding='valid', name='maxpool1')(y)
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

        super().__init__(model=model, folder=folder, model_name='SB_CNN')

    def sub_model():
        # example code on how define a new model based on the original
        new_model = Model(inputs=self.model.input, outputs=self.model.get_layer('dense1').output)
        return new_model


    # def train(...):  # i.e if want to redefine train function