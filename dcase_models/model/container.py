import numpy as np
import os
import json

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import model_from_json
import keras.backend as K

from ..utils.files import save_json
from ..utils.metrics import evaluate_metrics
from ..utils.callbacks import AccuracyCallback, F1ERCallback


class DCASEModelContainer():
    """
    A class that contain a keras model, the methods to train, evaluate,
    save and load the model. Child of this class can be created for
    especific models (i.e see SB_CNN class)

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
    def __init__(self, model=None, folder=None, model_name="APNet", metrics=['accuracy'] ,**kwargs):    
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
            self.load_model_from_json(folder, **kwargs)
        else:
            raise AttributeError("model or folder are both None")
        self.model_name = model_name
        self.metrics = metrics

    def train(self, X_train, Y_train, X_val, Y_val, weights_path= './',
               optimizer = 'Adam',learning_rate = 0.001, early_stopping=100, considered_improvement=0.01,
               losses='categorical_crossentropy', loss_weights=[1], sequence_time_sec=0.5, metric_resolution_sec=1.0, **kwargs_keras_fit):
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

        self.model.compile(loss=losses, optimizer=opt, loss_weights=loss_weights)

        file_weights = os.path.join(weights_path, 'best_weights.hdf5') 
        file_log = os.path.join(weights_path, 'training.log') 
        if self.metrics[0] == 'accuracy':
            metrics_callback = AccuracyCallback(X_val, Y_val, file_weights=file_weights, early_stopping=early_stopping, 
                                                considered_improvement=considered_improvement)
        if 'F1' in self.metrics: 
            metrics_callback = F1ERCallback(X_val, Y_val, file_weights=file_weights, early_stopping=early_stopping, 
                                            considered_improvement=considered_improvement, sequence_time_sec=sequence_time_sec, 
                                            metric_resolution_sec=metric_resolution_sec)
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
        return evaluate_metrics(self.model, X_test, Y_test, self.metrics)

    def load_model_from_json(self, folder, **kwargs):
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
        self.model = model_from_json(data, **kwargs)
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

