from dcase_models.util.files import save_json, save_pickle, load_pickle
from dcase_models.util.metrics import evaluate_metrics
from dcase_models.util.callbacks import ClassificationCallback, SEDCallback, TaggingCallback
from dcase_models.util.callbacks import PyTorchCallback
from dcase_models.data.data_generator import DataGenerator, KerasDataGenerator, PyTorchDataGenerator

import numpy as np
import os
import json
import inspect

from dcase_models.backend import backends

if 'torch' in backends:
    import torch
    from torch import nn

if ('tensorflow1' in backends) | ('tensorflow2' in backends):
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

if 'sklearn' in backends:
    import sklearn

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


class PyTorchModelContainer(ModelContainer):
    """ ModelContainer for pytorch models.

    A class that contains a pytorch model, the methods to train, evaluate,
    save and load the model. Descendants of this class can be specialized for
    specific models (i.e see SB_CNN class)

    Parameters
    ----------
    model : nn.Module or None, default=None
        If model is None the model is created with `build()`.

    model_path : str or None, default=None
        Path to the model. If it is not None, the model loaded from this path.

    model_name : str, default=PyTorchModelContainer
        Model name.

    metrics : list of str, default=['classification']
        List of metrics used for evaluation.
        See `dcase_models.utils.metrics`.

    kwargs
        Additional keyword arguments to `load_model_from_json()`.

    """

    if 'torch' in backends:
        class Model(nn.Module):
            # Define your model here
            def __init__(self, params):
                super().__init__()

            def forward(self, x):
                pass

    def __init__(self, model=None,
                 model_name="PyTorchModelContainer",
                 metrics=['classification'], use_cuda=True, **kwargs):

        if 'torch' not in backends:
            raise ImportError('Pytorch is not installed')

        self.use_cuda = use_cuda & torch.cuda.is_available()

        super().__init__(model=model, model_path=None,
                         model_name=model_name,
                         metrics=metrics)

        # Build the model
        if model is None:
            self.build()

    def build(self):
        """
        Define your model here
        """
        self.model = self.Model(self)

    def train(self, data_train, data_val, weights_path='./',
              optimizer='Adam', learning_rate=0.001, early_stopping=100,
              considered_improvement=0.01, losses='BCELoss',
              loss_weights=[1], batch_size=32, sequence_time_sec=0.5,
              metric_resolution_sec=1.0, label_list=[],
              shuffle=True, epochs=10):
        """
        Trains the pytorch model using the data and paramaters of arguments.

        Parameters
        ----------
        data_train : tuple of ndarray or DataGenerator
            Tuple or DataGenerator of training.
            Example of tuple: (X_train, Y_train) whose shapes
            are  (N_instances, N_hops, N_mel_bands) and (N_instances, N_classes) respectively.
            Example of DataGenerator: data_gen = DataGenerator(..., train=True)
        data_val : tuple of ndarray or DataGenerator
            Idem for validation set.
        weights_path : str
            Path where to save the best weights of the model
            in the training process
        optimizer : str or torch.nn.Optimizer
            Optimizer used to train the model. String argument should coincide
            with the class name in torch.optim
        learning_rate : float
            Learning rate used to train the model
        early_stopping : int
            Number of epochs to stop the training if there is not improvement
        considered_improvement : float
            Improvement in the performance metric considered to save a checkpoint.
        losses : (list of) torch loss functions (see https://pytorch.org/docs/stable/nn.html#loss-functions)
            Loss function(s) used for training.
        loss_weights : list
            List of weights for each loss function. Should be of the same length than losses
        batch_size : int
            Batch size used in the training process. Ignore if data_train is a DataGenerator
        sequence_time_sec : float
            Used for SED evaluation. Time resolution (in seconds) of the output
            (i.e features.sequence_hop_time)
        metric_resolution_sec: float
            Used for SED evaluation. Time resolution (in seconds) of evaluation
        label_list: list
            List of class labels (i.e dataset.label_list).
            This is needed for model evaluation.
        shuffle : bool
            If true the data_train is shuffle after each epoch.
            Ignored if data_train is DataGenerator
        epochs : int
            Number of training epochs

        """

        if type(losses) is not list:
            losses = [losses]

        for j, loss in enumerate(losses):
            if type(loss) is str:
                try:
                    loss_fn = getattr(torch.nn.modules.loss, loss)
                except:
                    raise AttributeError(
                        ("Loss {} not availabe. See the list of losses at "
                         "https://pytorch.org/docs/stable/nn.html#loss-functions").format(loss)
                    )
            else:
                if (torch.nn.modules.loss._Loss in inspect.getmro(loss.__class__)):
                    loss_fn = loss
                else:
                    raise AttributeError('loss should be a string or torch.nn.modules.loss._Loss')
            losses[j] = loss_fn()

        if type(loss_weights) is not list:
            loss_weights = [loss_weights]

        if len(loss_weights) != len(losses):
            raise AttributeError(
                ("loss_weights and losses should have the same length. Received: lengths {:d} and {:d} "
                 "respectively").format(len(loss_weights), len(losses))
            )

        if type(optimizer) is str:
            try:
                optimizer_function = getattr(torch.optim, optimizer)
            except:
                raise AttributeError(
                    ("Optimizer {} not availabe. See the list of optimizers at "
                     "https://pytorch.org/docs/stable/optim.html").format(optimizer)
                )
        else:
            if (torch.optim.optimizer.Optimizer in inspect.getmro(optimizer.__class__)):
                optimizer_function = optimizer
            else:
                raise AttributeError('optimizer should be a string or torch.optim.Optimizer')

        opt = optimizer_function(self.model.parameters(), lr=learning_rate)

        if self.use_cuda:
            self.model.to('cuda')

        train_from_numpy = False
        if type(data_train) is tuple:
            train_from_numpy = True
            X_train, Y_train = data_train
            if type(X_train) is not list:
                X_train = [X_train]
            if type(Y_train) is not list:
                Y_train = [Y_train]

            tensors_X = []
            tensors_Y = []
            for j in range(len(X_train)):
                tensors_X.append(torch.Tensor(X_train[j]))
                tensors_Y.append(torch.Tensor(Y_train[j]))

            torch_dataset = torch.utils.data.TensorDataset(*(tensors_X + tensors_Y))
            data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)

            n_inputs = len(tensors_X)
        else:
            torch_data_train = PyTorchDataGenerator(data_train)
            data_loader = torch.utils.data.DataLoader(torch_data_train, batch_size=1)

        current_metrics = [0]
        best_metrics = -np.inf
        epoch_best = 0
        epochs_since_improvement = 0

        if self.metrics[0] == 'sed':
            callback = SEDCallback(
                data_val, best_F1=-np.Inf, early_stopping=early_stopping, file_weights=weights_path,
                considered_improvement=considered_improvement, sequence_time_sec=sequence_time_sec,
                metric_resolution_sec=metric_resolution_sec, label_list=label_list
            )
        elif self.metrics[0] == 'classification':
            callback = ClassificationCallback(
                data_val, best_acc=-np.Inf, early_stopping=early_stopping, file_weights=weights_path,
                considered_improvement=considered_improvement,
                label_list=label_list
            )
        elif self.metrics[0] == 'tagging':
            callback = TaggingCallback(
                data_val, best_F1=-np.Inf, early_stopping=early_stopping, file_weights=weights_path,
                considered_improvement=considered_improvement, label_list=label_list
            )
        else:
            raise AttributeError("{} metric is not allowed".format(self.metrics[0]))

        callback = PyTorchCallback(self, callback)

        for epoch in range(epochs):
            # train
            for batch_ix, batch in enumerate(data_loader):
                # Compute prediction and loss
                if train_from_numpy:
                    X = batch[:n_inputs]
                    Y = batch[n_inputs:]
                else:
                    X, Y = batch
                    for j in range(len(X)):
                        X[j] = torch.squeeze(X[j], axis=0)
                        Y[j] = torch.squeeze(Y[j], axis=0)
                if self.use_cuda:
                    for j in range(len(X)):
                        X[j] = X[j].cuda()
                pred = self.model(*X)
                if type(pred) is not list:
                    preds = [pred]
                else:
                    preds = pred
                assert len(preds) == len(losses)
                loss = 0
                for loss_fn, loss_weight, pred, gt in zip(losses, loss_weights, preds, Y):
                    if self.use_cuda:
                        gt = gt.cuda()
                    loss += loss_weight*loss_fn(pred.float(), gt.float())

                # Backpropagation
                opt.zero_grad()
                loss.backward()
                opt.step()

                if batch == len(data_train) - 1:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{len(data_train):>5d}]")

            if shuffle & (not train_from_numpy):
                torch_data_train.shuffle_list()

            # validation
            with torch.no_grad():
                callback.on_epoch_end(epoch)
                if callback.stop_training:
                    break

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
        return evaluate_metrics(self, data_test, self.metrics, **kwargs)

    def load_model_from_json(self, folder, **kwargs):
        """
        Loads a model from a model.json file in the path given by folder.
        The model is load in self.model attribute.

        Parameters
        ----------
        folder : str
            Path to the folder that contains model.json file
        """
        raise NotImplementedError()

    def save_model_json(self, folder):
        """
        Saves the model to a model.json file in the given folder path.

        Parameters
        ----------
        folder : str
            Path to the folder to save model.json file
        """
        raise NotImplementedError()

    def save_model_weights(self, weights_folder):
        """
        Saves self.model weights in weights_folder/best_weights.hdf5.

        Parameters
        ----------
        weights_folder : str
            Path to save the weights file
        """
        weights_file = 'best_weights.pth'
        weights_path = os.path.join(weights_folder, weights_file)
        torch.save(self.model.state_dict(), weights_path)

    def load_model_weights(self, weights_folder):
        """
        Loads self.model weights in weights_folder/best_weights.hdf5.

        Parameters
        ----------
        weights_folder : str
            Path to save the weights file.

        """
        weights_file = 'best_weights.pth'
        weights_path = os.path.join(weights_folder, weights_file)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def load_pretrained_model_weights(self,
                                      weights_folder='./pretrained_weights'):
        """
        Loads pretrained weights to self.model weights.

        Parameters
        ----------
        weights_folder : str
            Path to load the weights file

        """
        raise NotImplementedError()

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def check_if_model_exists(self, folder, **kwargs):
        """ Checks if the model already exits in the path.

        Check if the folder/model.json file exists and includes
        the same model as self.model.

        Parameters
        ----------
        folder : str
            Path to the folder to check.

        """
        weights_file = 'best_weights.pth'
        weights_path = os.path.join(folder, weights_file)
        new_model = self.Model(self)
        try:
            new_model.load_state_dict(torch.load(weights_path))
        except:
            return False

        return True

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_available_intermediate_outputs(self):
        """ Return a list of available intermediate outputs.

        Return a list of model's layers.

        Returns
        -------
        list of str
            List of layers names.

        """
        raise NotImplementedError()

    def get_intermediate_output(self, output_ix_name, inputs):
        """ Return the output of the model in a given layer.

        Cut the model in the given layer and predict the output
        for the given inputs.

        Returns
        -------
        ndarray
            Output of the model in the given layer.

        """
        raise NotImplementedError()

    def predict(self, x):
        # Imitate keras.predict() function
        """ Imitate the output of the keras.predict() function.

        Cut the model in the given layer and predict the output
        for the given inputs.

        Parameters
        ----------
        x: (list of) ndarray
            Model's input(s)

        Returns
        -------
        (list of) ndarray
            Model's input(s)

        """
        if type(x) is not list:
            x = [x]
        for j in range(len(x)):
            x[j] = torch.tensor(x[j].astype(float), dtype=torch.float)
            if self.use_cuda:
                x[j] = x[j].cuda()
        if self.use_cuda:
            self.model.cuda()

        y = self.model(*x)

        if (type(y) is list) or (type(y) is tuple):
            y_np = []
            for j in range(len(y)):
                y_np.append(y[j].cpu().detach().numpy())
        else:
            y_np = y.cpu().detach().numpy()

        return y_np


class SklearnModelContainer(ModelContainer):
    """ ModelContainer for scikit-learn models.

    A class that contains a scikit-learn classifier, the methods to train, evaluate,
    save and load the classifier.

    Parameters
    ----------
    model : scikit-learn model, default=None
        If model is None the model is loaded from model_path

    model_path : str or None, default=None
        Path to the model. If model is None, and model_path is not None, the model is loaded from this path.

    model_name : str, default=SklearnModelContainer
        Model name.

    metrics : list of str, default=['classification']
        List of metrics used for evaluation.
        See `dcase_models.utils.metrics`.

    """

    def __init__(self, model=None, model_path=None,
                 model_name="SklearnModelContainer",
                 metrics=['classification']):

        if (model is None) & (model_path is None):
            raise AttributeError("model or model_path should be passed as argument")

        super().__init__(model=model, model_path=None,
                         model_name=model_name,
                         metrics=metrics)

        if (model is None) & (model_path is not None):
            self.load_model_weights(model_path)

    def build(self):
        """
        Not used
        """
        raise NotImplementedError()

    def train(self, data_train, data_val=None, weights_path='./',
              sequence_time_sec=0.5, metric_resolution_sec=1.0, label_list=[],
              **kwargs):
        """
        Trains the scikit-learn model using the data and parameters of arguments.

        Parameters
        ----------
        data_train : tuple of ndarray or DataGenerator
            Tuple or DataGenerator of training.
            Tuple should include inputs and ouputs for training: (X_train, Y_train) whose shapes
            are for instances (N_instances, N_mel_bands) and (N_instances, N_classes) or (N_instances,) respectively.
            Example of DataGenerator: data_gen = DataGenerator(..., train=True)
        data_val : None or tuple of ndarray or DataGenerator
            Idem for validation set. If None, there is no validation when the training is ended.
        weights_path : str
            Path where to save the best weights of the model after the training process
        sequence_time_sec : float
            Used for SED evaluation. Time resolution (in seconds) of the output. Ignored if data_val is None.
            (i.e features.sequence_hop_time)
        metric_resolution_sec: float
            Used for SED evaluation. Time resolution (in seconds) of evaluation. Ignored if data_val is None.
        label_list: list
            List of class labels (i.e dataset.label_list). Ignored if data_val is None.
            This is needed for model evaluation.
        kwargs: kwargs
            kwargs of sklearn's fit() function

        """
        if type(data_train) is not tuple:
            # DataGenerator, check if partial_fit is available
            if 'partial_fit' not in dir(self.model):
                raise AttributeError(
                    ("This model does not allow partial_fit, and therefore data_train should be a numpy array. "
                     "Please call DataGenerator.get_data() before."))
            for batch in range(len(data_train)):
                X, Y = data_train.get_data_batch(batch)
                if type(X) is not np.ndarray:
                    raise AttributeError("Multi-input is not allowed")
                if type(Y) is not np.ndarray:
                    raise AttributeError("Multi-output is not allowed")
                if len(X.shape) != 2:
                    raise AttributeError("The input should be a 2D array. Received shape {}".format(X.shape))
                if len(Y.shape) > 2:
                    raise AttributeError("The output should be a 1D or 2D array. Received shape {}".format(Y.shape))
                classes = np.arange(len(label_list))
                self.model.partial_fit(X, Y, classes)
        else:
            if type(data_train[0]) is not np.ndarray:
                raise AttributeError("Multi-input is not allowed")
            if type(data_train[1]) is not np.ndarray:
                raise AttributeError("Multi-output is not allowed")
            if len(data_train[0].shape) != 2:
                raise AttributeError("The input should be a 2D array. Received shape {}".format(data_train[0].shape))
            if len(data_train[1].shape) > 2:
                raise AttributeError(
                    "The output should be a 1D or 2D array. Received shape {}".format(data_train[1].shape))

            self.model.fit(data_train[0], data_train[1], **kwargs)

        self.save_model_weights(weights_path)

        kwargs = {}
        if self.metrics[0] == 'sed':
            kwargs = {
                'sequence_time_sec': sequence_time_sec,
                'metric_resolution_sec': metric_resolution_sec
            }
        if data_val is not None:
            results = evaluate_metrics(
                self,
                data_val,
                self.metrics,
                label_list=label_list,
                **kwargs
            )
            return results[self.metrics[0]]

    def evaluate(self, data_test, **kwargs):
        """
        Evaluates the keras model using X_test and Y_test.

        Parameters
        ----------
        X_test : ndarray
            2D array with mel-spectrograms of test set.
            Shape = (N_instances, N_mel_bands)
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
        return evaluate_metrics(self, data_test, self.metrics, **kwargs)

    def load_model_from_json(self, folder, **kwargs):
        """
        Loads a model from a model.json file in the path given by folder.
        The model is load in self.model attribute.

        Parameters
        ----------
        folder : str
            Path to the folder that contains model.json file
        """
        raise NotImplementedError()

    def save_model_json(self, folder):
        """
        Saves the model to a model.json file in the given folder path.

        Parameters
        ----------
        folder : str
            Path to the folder to save model.json file
        """
        raise NotImplementedError()

    def save_model_weights(self, weights_folder):
        """
        Saves self.model weights in weights_folder/best_weights.hdf5.

        Parameters
        ----------
        weights_folder : str
            Path to save the weights file
        """
        weights_file = 'model.skl'
        weights_path = os.path.join(weights_folder, weights_file)
        save_pickle(self.model, weights_path)

    def load_model_weights(self, weights_folder):
        """
        Loads self.model weights in weights_folder/best_weights.hdf5.

        Parameters
        ----------
        weights_folder : str
            Path to save the weights file.

        """
        weights_file = 'model.skl'
        weights_path = os.path.join(weights_folder, weights_file)
        self.model = load_pickle(weights_path)

    def load_pretrained_model_weights(self,
                                      weights_folder='./pretrained_weights'):
        raise NotImplementedError()

    def get_number_of_parameters(self):
        return len(self.model.get_params())

    def check_if_model_exists(self, folder, **kwargs):
        """ Checks if the model already exits in the path.

        Check if the folder/model.json file exists and includes
        the same model as self.model.

        Parameters
        ----------
        folder : str
            Path to the folder to check.

        """
        weights_file = 'model.skl'
        weights_path = os.path.join(folder, weights_file)
        if not os.path.exists(weights_path):
            return False

        new_model = load_pickle(weights_path)

        if new_model.__class__.__name__ != self.model.__class__.__name__:
            return False

        new_params = new_model.get_params()
        for key, value in self.model.get_params().items():
            if value != new_params[key]:
                return False
        return True

    def cut_network(self, layer_where_to_cut):
        raise NotImplementedError()

    def fine_tuning(self, layer_where_to_cut, new_number_of_classes=10,
                    new_activation='softmax',
                    freeze_source_model=True, new_model=None):
        raise NotImplementedError()

    def get_available_intermediate_outputs(self):
        raise NotImplementedError()

    def get_intermediate_output(self, output_ix_name, inputs):
        raise NotImplementedError()

    def predict(self, x):
        # Imitate keras.predict() function
        """ Imitates the output of the keras.predict() function.

        Parameters
        ----------
        x: ndarray
            Model's input

        Returns
        -------
        ndarray
            Model's output

        """
        pred = self.model.predict(x)
        if len(pred.shape) == 1:
            # if single output, apply one-hot encoder (needed for evaluation)
            y = np.zeros((len(pred), len(self.model.classes_)))
            for j in range(len(pred)):
                y[j, int(pred[j])] = 1
        else:
            y = pred
        return y
