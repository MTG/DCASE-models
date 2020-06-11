import numpy as np
import os
import json

import keras.backend as K
from keras.callbacks import CSVLogger
from keras.models import model_from_json, Model
from keras.layers import Dense, Input

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

    def __init__(self, model=None, model_path=None,
                 model_name="DCASEModelContainer",
                 metrics=['accuracy'], **kwargs):
        """
        Parameters
        ----------
        model : keras model
            APNet keras model (see apnet.model.apnet())
        folder : str
            if model is None, this path is used to load the model.
            See load_model_from_json()
        """
        self.model = model
        self.model_path = model_path
        self.model_name = model_name
        self.metrics = metrics
        self.kwargs = kwargs

        # Build the model
        self.build()

    def build(self):
        # Load model from model_path if the model is not defined
        if (self.model is None) and (self.model_name is not None):
            self.load_model_from_json(self.model_path, **self.kwargs)

    def train(self, X_train, Y_train, X_val, Y_val, weights_path='./',
              optimizer='Adam', learning_rate=0.001, early_stopping=100,
              considered_improvement=0.01, losses='categorical_crossentropy',
              loss_weights=[1], sequence_time_sec=0.5,
              metric_resolution_sec=1.0, **kwargs_keras_fit):
        """
        Train the keras model using the data and paramaters of arguments.

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
            Path where to save the best weights of the model
            in the training process
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

        self.model.compile(loss=losses, optimizer=opt,
                           loss_weights=loss_weights)

        file_weights = os.path.join(weights_path, 'best_weights.hdf5')
        file_log = os.path.join(weights_path, 'training.log')
        if self.metrics[0] == 'accuracy':
            metrics_callback = AccuracyCallback(
                X_val, Y_val, file_weights=file_weights,
                early_stopping=early_stopping,
                considered_improvement=considered_improvement
            )
        if 'F1' in self.metrics:
            metrics_callback = F1ERCallback(
                X_val, Y_val, file_weights=file_weights,
                early_stopping=early_stopping,
                considered_improvement=considered_improvement,
                sequence_time_sec=sequence_time_sec,
                metric_resolution_sec=metric_resolution_sec
            )
        log = CSVLogger(file_log)
        self.model.fit(x=X_train, y=Y_train, shuffle=True,
                       callbacks=[metrics_callback, log],
                       **kwargs_keras_fit)

    def evaluate(self, X_test, Y_test, scaler=None):
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
        # weights_file = os.path.join(folder, 'best_weights.hdf5')
        json_file = os.path.join(folder, 'model.json')

        with open(json_file) as json_f:
            data = json.load(json_f)
        self.model = model_from_json(data, **kwargs)
        # self.model.load_weights(weights_file)

    def save_model_json(self, folder):
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
        Load self.model weights in weights_folder/best_weights.hdf5.

        Parameters
        ----------
        weights_folder : str
            Path to save the weights file
        """
        weights_file = 'best_weights.hdf5'
        weights_path = os.path.join(weights_folder, weights_file)
        self.model.load_weights(weights_path)

    def load_pretrained_model_weights(self,
                                      weights_folder='./pretrained_weights'):
        """
        Load pretrained weights to self.model weights

        Parameters
        ----------
        weights_folder : str
            Path to load the weights file
        """
        basepath = os.path.dirname(__file__)
        weights_file = self.model_name + '.hdf5'
        weights_path = os.path.join(basepath, weights_folder, weights_file)
        self.model.load_weights(weights_path, by_name=True)

    def get_numer_of_parameters(self):
        trainable_count = int(
            np.sum([K.count_params(p) for p in
                    set(self.model.trainable_weights)]))
        return trainable_count

    def check_if_model_exists(self, folder, **kwargs):
        """
        Save model parameters to parameters.json file in
        the path given by argument.

        Parameters
        ----------
        folder : str
            Path to the folder to save model.json file
        """
        json_file = os.path.join(folder, 'model.json')
        if not os.path.exists(json_file):
            return False

        with open(json_file) as json_f:
            data = json.load(json_f)
        model_saved = model_from_json(data, **kwargs)

        models_are_same = True
        self.model.summary()
        model_saved.summary()

        for l1, l2 in zip(self.model.layers, model_saved.layers):
            print(l1.get_config() == l2.get_config())
            if l1.get_config() != l2.get_config():

                models_are_same = False
                break

        # print(type(json_model))
        # print(type(json_saved))
        # print(json_model==json_saved)
        print(models_are_same)
        return models_are_same

    def cut_network(self, layer_where_to_cut):
        if type(layer_where_to_cut) == str:
            last_layer = self.model.get_layer(layer_where_to_cut)
        elif type(layer_where_to_cut) == int:
            last_layer = self.model.layers[layer_where_to_cut]
        else:
            raise AttributeError(
                "layer_where_to_cut has to be str or int type")
        model_without_last_layer = Model(
            self.model.input, last_layer.output, name='source_model')

        return model_without_last_layer

    def fine_tuning(self, layer_where_to_cut, new_number_of_classes=10,
                    new_activation='softmax',
                    freeze_source_model=True, new_model=None):
        """
        Create a new model for fine-tuning. Cut the model in
        the layer_where_to_cut layer
        and add a new fully-connected layer.

        Parameters
        ----------
        layer_where_to_cut : str or int
            Name (str) of index (int) of the layer where cut the model.
            This layer is included in the new model.

        new_number_of_classes : int
            Number of units in the new fully-connected layer
            (number of classes)

        new_activation : str
            Activitation of the new fully-connected layer

        freeze_source_model : bool
            If True, the source model is set to not be trainable

        new_model : Keras Model
            If is not None, this model is add after the cut model.
            This is useful if you want add more than
            a fully-connected layer.
        """
        # cut last layer
        model_without_last_layer = self.cut_network(layer_where_to_cut)

        # add a new fully connected layer
        input_shape = self.model.layers[0].output_shape[1:]
        x = Input(shape=input_shape, dtype='float32', name='input')
        y = model_without_last_layer(x)

        if new_model is None:
            y = Dense(new_number_of_classes,
                      activation=new_activation, name='new_dense_layer')(y)
        else:
            y = new_model(y)

        # change self.model with fine_tuned model
        self.model = Model(x, y)

        # freeze the source model if freeze_source_model is True
        self.model.get_layer(
            'source_model').trainable = not freeze_source_model
