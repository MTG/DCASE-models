import numpy as np
import os
import json

import tensorflow as tf
tensorflow2 = tf.__version__.split('.')[0] == '2'

if tensorflow2:
    import tensorflow.keras.backend as K
    from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
    from tensorflow.keras.models import model_from_json, Model
    from tensorflow.keras.layers import Dense, Input
else:
    import keras.backend as K
    from keras.callbacks import CSVLogger, ModelCheckpoint
    from keras.models import model_from_json, Model
    from keras.layers import Dense, Input 

from dcase_models.util.files import save_json
from dcase_models.util.metrics import evaluate_metrics
from dcase_models.util.callbacks import ClassificationCallback, SEDCallback
from dcase_models.util.callbacks import TaggingCallback
from dcase_models.data.data_generator import DataGenerator, KerasDataGenerator


class ModelContainer():
    """ Abstract base class to store and manage models.

    Parameters
    ----------
    model : keras model or similar
        Object that defines the model (i.e keras.models.Model)
    model_path : str
        Path to the model file
    model_name : str
        Model name
    metrics : list of str
        List of metrics used for evaluation

    """

    def __init__(self, model=None, model_path=None,
                 model_name="ModelContainer",
                 metrics=['classification']):
        """ Initialize ModelContainer

        Parameters
        ----------
        model : keras model or similar
            Object that defines the model (i.e keras.models.Model)
        model_path : str
            Path to the model file
        model_name : str
            Model name
        metrics : list of str
            List of metrics used for evaluation

        """
        self.model = model
        self.model_path = model_path
        self.model_name = model_name
        self.metrics = metrics

    def build(self):
        """ Missing docstring here
        """
        raise NotImplementedError

    def train(self):
        """ Missing docstring here
        """
        raise NotImplementedError

    def evaluate(self, X_test, Y_test, scaler=None):
        """ Missing docstring here
        """
        raise NotImplementedError

    def save_model_json(self, folder):
        """ Missing docstring here
        """
        raise NotImplementedError

    def load_model_from_json(self, folder, **kwargs):
        """ Missing docstring here
        """
        raise NotImplementedError

    def save_model_weights(self, weights_folder):
        """ Missing docstring here
        """
        raise NotImplementedError

    def load_model_weights(self, weights_folder):
        """ Missing docstring here
        """
        raise NotImplementedError

    def get_number_of_parameters(self):
        """ Missing docstring here
        """
        raise NotImplementedError

    def check_if_model_exists(self, folder, **kwargs):
        """ Missing docstring here
        """
        raise NotImplementedError

    def get_available_intermediate_outputs(self):
        """ Missing docstring here
        """
        raise NotImplementedError

    def get_intermediate_output(self, output_ix_name):
        """ Missing docstring here
        """
        raise NotImplementedError


class KerasModelContainer(ModelContainer):
    """ ModelContainer for keras models.

    A class that contains a keras model, the methods to train, evaluate,
    save and load the model. Descendants of this class can be specialized for
    specific models (i.e see SB_CNN class)

    Parameters
    ----------
    model : keras.models.Model or None, default=None
        If model is None the model is created with `build()`.

    model_path : str or None, default=None
        Path to the model. If it is not None, the model loaded from this path.

    model_name : str, default=DCASEModelContainer
        Model name.

    metrics : list of str, default=['classification']
        List of metrics used for evaluation.
        See `dcase_models.utils.metrics`.

    kwargs
        Additional keyword arguments to `load_model_from_json()`.

    """

    def __init__(self, model=None, model_path=None,
                 model_name="DCASEModelContainer",
                 metrics=['classification'], **kwargs):
        super().__init__(model=model, model_path=model_path,
                         model_name=model_name,
                         metrics=metrics)

        # Build or load the model
        if self.model_path is None:
            self.build()
        else:
            self.load_model_from_json(self.model_path, **kwargs)

    def build(self):
        """
        Define your model here
        """
        pass

    def train(self, data_train, data_val, weights_path='./',
              optimizer='Adam', learning_rate=0.001, early_stopping=100,
              considered_improvement=0.01, losses='categorical_crossentropy',
              loss_weights=[1], sequence_time_sec=0.5,
              metric_resolution_sec=1.0, label_list=[],
              shuffle=True, **kwargs_keras_fit):
        """
        Trains the keras model using the data and paramaters of arguments.

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
            Number of training epochs
        fit_verbose : int
            Verbose mode for fit method of Keras model

        """
        if tensorflow2:
            import tensorflow.keras.optimizers as optimizers
        else:
            import keras.optimizers as optimizers
        optimizer_function = getattr(optimizers, optimizer)
        opt = optimizer_function(lr=learning_rate)

        self.model.compile(loss=losses, optimizer=opt,
                           loss_weights=loss_weights)

        file_weights = os.path.join(weights_path, 'best_weights.hdf5')
        file_log = os.path.join(weights_path, 'training.log')

        if self.metrics[0] == 'classification':
            metrics_callback = ClassificationCallback(
                data_val, file_weights=file_weights,
                early_stopping=early_stopping,
                considered_improvement=considered_improvement,
                label_list=label_list
            )
        elif self.metrics[0] == 'sed':
            metrics_callback = SEDCallback(
                data_val, file_weights=file_weights,
                early_stopping=early_stopping,
                considered_improvement=considered_improvement,
                sequence_time_sec=sequence_time_sec,
                metric_resolution_sec=metric_resolution_sec,
                label_list=label_list
            )
        elif self.metrics[0] == 'tagging':
            metrics_callback = TaggingCallback(
                data_val, file_weights=file_weights,
                early_stopping=early_stopping,
                considered_improvement=considered_improvement,
                label_list=label_list
            )
        else:
            metrics_callback = ModelCheckpoint(
                filepath=file_weights,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1)

        log = CSVLogger(file_log)

        validation_data = None
        if metrics_callback.__class__ is ModelCheckpoint:
            if data_val.__class__ is DataGenerator:
                validation_data = KerasDataGenerator(data_val)
            else:
                validation_data = data_val
        if type(data_train) in [list, tuple]:
            self.model.fit(
                x=data_train[0], y=data_train[1], shuffle=shuffle,
                callbacks=[metrics_callback, log],
                validation_data=validation_data,
                **kwargs_keras_fit
            )
        else:
            if data_train.__class__ is DataGenerator:
                data_train = KerasDataGenerator(data_train)
            kwargs_keras_fit.pop('batch_size')
            self.model.fit_generator(
                generator=data_train,
                callbacks=[metrics_callback, log],
                validation_data=validation_data,
                **kwargs_keras_fit
                # use_multiprocessing=True,
                # workers=6)
            )

    def evaluate(self, data_test, **kwargs):
        """
        Evaluates the keras model using X_test and Y_test.

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
            evaluation's accuracy
        list
            list of annotations (ground_truth)
        list
            list of model predictions

        """
        return evaluate_metrics(
            self.model, data_test, self.metrics, **kwargs
        )

    def load_model_from_json(self, folder, **kwargs):
        """
        Loads a model from a model.json file in the path given by folder.
        The model is load in self.model attribute.

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
        Saves the model to a model.json file in the given folder path.

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
        Saves self.model weights in weights_folder/best_weights.hdf5.

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
        Loads self.model weights in weights_folder/best_weights.hdf5.

        Parameters
        ----------
        weights_folder : str
            Path to save the weights file.

        """
        weights_file = 'best_weights.hdf5'
        weights_path = os.path.join(weights_folder, weights_file)
        self.model.load_weights(weights_path)

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
        self.model.load_weights(weights_path, by_name=True)

    def get_number_of_parameters(self):
        trainable_count = int(
            np.sum([K.count_params(p) for p in
                    set(self.model.trainable_weights)]))
        return trainable_count

    def check_if_model_exists(self, folder, **kwargs):
        """ Checks if the model already exits in the path.

        Check if the folder/model.json file exists and includes
        the same model as self.model.

        Parameters
        ----------
        folder : str
            Path to the folder to check.

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
            if l1.get_config() != l2.get_config():
                models_are_same = False
                break

        return models_are_same

    def cut_network(self, layer_where_to_cut):
        """ Cuts the network at the layer passed as argument.

        Parameters
        ----------
        layer_where_to_cut : str or int
            Layer name (str) or index (int) where cut the model.

        Returns
        -------
        keras.models.Model
            Cutted model.

        """
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
        """ Create a new model for fine-tuning.

        Cut the model in the layer_where_to_cut layer
        and add a new fully-connected layer.

        Parameters
        ----------
        layer_where_to_cut : str or int
            Name (str) of index (int) of the layer where cut the model.
            This layer is included in the new model.

        new_number_of_classes : int
            Number of units in the new fully-connected layer
            (number of classes).

        new_activation : str
            Activation of the new fully-connected layer.

        freeze_source_model : bool
            If True, the source model is set to not be trainable.

        new_model : Keras Model
            If is not None, this model is added after the cut model.
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

    def get_available_intermediate_outputs(self):
        """ Return a list of available intermediate outputs.

        Return a list of model's layers.

        Returns
        -------
        list of str
            List of layers names.

        """
        layer_names = [layer.name for layer in self.model.layers]
        return layer_names

    def get_intermediate_output(self, output_ix_name, inputs):
        """ Return the output of the model in a given layer.

        Cut the model in the given layer and predict the output
        for the given inputs.

        Returns
        -------
        ndarray
            Output of the model in the given layer.

        """
        if type(output_ix_name) == int:
            cut_model = self.cut_network(output_ix_name)
            output = cut_model.predict(inputs)
        else:
            if output_ix_name in self.get_available_intermediate_outputs():
                cut_model = self.cut_network(output_ix_name)
                output = cut_model.predict(inputs)
            else:
                return None

        return output
