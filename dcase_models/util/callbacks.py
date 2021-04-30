# encoding: utf-8
"""Callback functions"""

from dcase_models.util.metrics import evaluate_metrics

import tensorflow as tf
tensorflow2 = tf.__version__.split('.')[0] == '2'

if tensorflow2:
    from tensorflow.keras.callbacks import Callback
else:
    from keras.callbacks import Callback

eps = 1e-6


class ClassificationCallback(Callback):
    """Keras callback to calculate acc after each epoch and save
    file with the weights if the evaluation improves
    """

    def __init__(self, data, file_weights=None, best_acc=0,
                 early_stopping=0, considered_improvement=0.01,
                 label_list=[]):
        """ Initialize the keras callback

        Parameters
        ----------
        data : tuple or KerasDataGenerator
            Validation data for model evaluation
            (X_val, Y_val) or KerasDataGenerator

        file_weights : string
            Path to the file with the weights

        best_acc : float
            Last accuracy value, only if continue

        early_stopping : int
            Number of epochs for cut the training if not improves
            if 0, do not use it
        """

        self.data = data
        self.best_acc = best_acc
        self.file_weights = file_weights
        self.early_stopping = early_stopping
        self.epochs_since_improvement = 0
        self.epoch_best = 0
        self.considered_improvement = considered_improvement
        self.label_list = label_list

    def on_epoch_end(self, epoch, logs={}):
        """ This function is run when each epoch ends.
        The metrics are calculated, printed and saved to the log file.

        Parameters
        ----------
        epoch : int
            number of epoch (from Callback class)

        logs : dict
            log data (from Callback class)

        """
        results = evaluate_metrics(
           self.model, self.data, ['classification'],
           label_list=self.label_list)

        results = results['classification'].results()
        acc = results['overall']['accuracy']
        logs['accuracy'] = acc

        self.current_acc = acc

        if self.current_acc > self.best_acc + self.considered_improvement:
            self.best_acc = self.current_acc
            self.model.save_weights(self.file_weights)
            msg = 'Acc = {:.4f} - Best val Acc: {:.4f} (IMPROVEMENT, saving)\n'
            print(msg.format(self.current_acc, self.best_acc))
            self.epochs_since_improvement = 0
            self.epoch_best = epoch
        else:
            print('Acc = {:.4f} - Best val Acc: {:.4f} ({:d})\n'.format(
                self.current_acc, self.best_acc, self.epoch_best))
            self.epochs_since_improvement += 1
        if self.epochs_since_improvement >= self.early_stopping-1:
            print('Not improvement for %d epochs, stopping the training' %
                  self.early_stopping)
            self.model.stop_training = True


class SEDCallback(Callback):
    """Keras callback to calculate F1 and ER after each epoch and save
    file with the weights if the evaluation improves.

    Use sed_eval library.
    """

    def __init__(self, data, file_weights=None, best_F1=0,
                 early_stopping=0, considered_improvement=0.01,
                 sequence_time_sec=0.5, metric_resolution_sec=1.0,
                 label_list=[]):
        """ Initialize the keras callback

        Parameters
        ----------
        data : tuple or KerasDataGenerator
            Validation data for model evaluation
            (X_val, Y_val) or KerasDataGenerator

        file_weights : string
            Path to the file with the weights

        best_acc : float
            Last accuracy value, only if continue

        early_stopping : int
            Number of epochs for cut the training if not improves
            if 0, do not use it
        """

        self.data = data
        self.best_F1 = best_F1
        self.file_weights = file_weights
        self.early_stopping = early_stopping
        self.sequence_time_sec = sequence_time_sec
        self.metric_resolution_sec = metric_resolution_sec
        self.epochs_since_improvement = 0
        self.epoch_best = 0
        self.considered_improvement = considered_improvement
        self.label_list = label_list

    def on_epoch_end(self, epoch, logs={}):
        """ This function is run when each epoch ends.
        The metrics are calculated, printed and saved to the log file.

        Parameters
        ----------
        epoch : int
            number of epoch (from Callback class)

        logs : dict
            log data (from Callback class)

        """
        results = evaluate_metrics(self.model,
                                   self.data, ['sed'],
                                   label_list=self.label_list)

        results = results['sed'].results()
        F1 = results['overall']['f_measure']['f_measure']
        ER = results['overall']['error_rate']['error_rate']
        logs['F1'] = F1
        logs['ER'] = ER

        self.current_F1 = F1

        if self.current_F1 > self.best_F1 + self.considered_improvement:
            self.best_F1 = self.current_F1
            self.model.save_weights(self.file_weights)
            msg = """F1 = {:.4f}, ER = {:.4f} - Best val F1: {:.4f}
                  (IMPROVEMENT, saving)\n"""
            print(msg.format(self.current_F1, ER, self.best_F1))
            self.epochs_since_improvement = 0
            self.epoch_best = epoch
        else:
            msg = 'F1 = {:.4f}, ER = {:.4f} - Best val F1: {:.4f} ({:d})\n'
            print(msg.format(self.current_F1, ER,
                             self.best_F1, self.epoch_best))
            self.epochs_since_improvement += 1
        if self.epochs_since_improvement >= self.early_stopping-1:
            print('Not improvement for %d epochs, stopping the training' %
                  self.early_stopping)
            self.model.stop_training = True


class TaggingCallback(Callback):
    """Keras callback to calculate acc after each epoch and save
    file with the weights if the evaluation improves
    """

    def __init__(self, data, file_weights=None, best_F1=0,
                 early_stopping=0, considered_improvement=0.01,
                 label_list=[]):
        """ Initialize the keras callback

        Parameters
        ----------
        data : tuple or KerasDataGenerator
            Validation data for model evaluation
            (X_val, Y_val) or KerasDataGenerator

        file_weights : string
            Path to the file with the weights

        best_acc : float
            Last accuracy value, only if continue

        early_stopping : int
            Number of epochs for cut the training if not improves
            if 0, do not use it
        """

        self.data = data
        self.best_F1 = best_F1
        self.file_weights = file_weights
        self.early_stopping = early_stopping
        self.epochs_since_improvement = 0
        self.epoch_best = 0
        self.considered_improvement = considered_improvement
        self.label_list = label_list

    def on_epoch_end(self, epoch, logs={}):
        """ This function is run when each epoch ends.
        The metrics are calculated, printed and saved to the log file.

        Parameters
        ----------
        epoch : int
            number of epoch (from Callback class)

        logs : dict
            log data (from Callback class)

        """
        results = evaluate_metrics(self.model,
                                   self.data, ['tagging'],
                                   label_list=self.label_list)

        results = results['tagging'].results()

        F1 = results['overall']['f_measure']['f_measure']
        logs['F1'] = F1

        self.current_F1 = F1

        if self.current_F1 > self.best_F1 + self.considered_improvement:
            self.best_F1 = self.current_F1
            self.model.save_weights(self.file_weights)
            msg = 'F1 = {:.4f} - Best val F1: {:.4f} (IMPROVEMENT, saving)\n'
            print(msg.format(self.current_F1, self.best_F1))
            self.epochs_since_improvement = 0
            self.epoch_best = epoch
        else:
            print('F1 = {:.4f} - Best val F1: {:.4f} ({:d})\n'.format(
                self.current_F1, self.best_F1, self.epoch_best))
            self.epochs_since_improvement += 1
        if self.epochs_since_improvement >= self.early_stopping-1:
            print('Not improvement for %d epochs, stopping the training' %
                  self.early_stopping)
            self.model.stop_training = True
