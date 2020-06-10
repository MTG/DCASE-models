from .metrics import evaluate_metrics
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import matplotlib
matplotlib.use('Agg')
eps = 1e-6


class AccuracyCallback(Callback):
    """Keras callback to calculate acc after each epoch and save 
    file with the weights if the evaluation improves
    """

    def __init__(self, X_val, Y_val, file_weights=None, best_acc=0, early_stopping=0, considered_improvement=0.01):
        """ Initialize the keras callback
        Parameters
        ----------
        X_val : array
            Validation data for model evaluation

        Y_val : array
            Ground-truth of th validation set

        file_weights : string
            Path to the file with the weights

        best_acc : float
            Last accuarcy value, only if continue 

        early_stopping : int
            Number of epochs for cut the training if not improves
            if 0, do not use it    
        """

        self.X_val = X_val
        self.Y_val = Y_val
        self.best_acc = best_acc
        self.file_weights = file_weights
        self.early_stopping = early_stopping
        self.epochs_since_improvement = 0
        self.epoch_best = 0
        self.considered_improvement = considered_improvement

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
            self.model, self.X_val, self.Y_val, ['accuracy'])
        acc = results['accuracy']
        logs['Acc'] = acc

        self.current_acc = acc

        if self.current_acc > self.best_acc + self.considered_improvement:
            self.best_acc = self.current_acc
            self.model.save_weights(self.file_weights)
            print('Acc = {:.4f} -  Best val Acc: {:.4f} (IMPROVEMENT, saving)\n'.format(
                self.current_acc, self.best_acc))
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


class F1ERCallback(Callback):
    """Keras callback to calculate F1 and ER after each epoch and save 
    file with the weights if the evaluation improves
    """

    def __init__(self, X_val, Y_val, file_weights=None, best_F1=0, early_stopping=0, considered_improvement=0.01, sequence_time_sec=0.5, metric_resolution_sec=1.0):
        """ Initialize the keras callback
        Parameters
        ----------
        X_val : array
            Validation data for model evaluation

        Y_val : array
            Ground-truth of th validation set

        file_weights : string
            Path to the file with the weights

        best_acc : float
            Last accuarcy value, only if continue 

        early_stopping : int
            Number of epochs for cut the training if not improves
            if 0, do not use it    
        """

        self.X_val = X_val
        self.Y_val = Y_val
        self.best_F1 = best_F1
        self.file_weights = file_weights
        self.early_stopping = early_stopping
        self.sequence_time_sec = sequence_time_sec
        self.metric_resolution_sec = metric_resolution_sec
        self.epochs_since_improvement = 0
        self.epoch_best = 0
        self.considered_improvement = considered_improvement

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
        results = evaluate_metrics(self.model, self.X_val, self.Y_val, ['F1', 'ER'],
                                   sequence_time_sec=self.sequence_time_sec, metric_resolution_sec=self.metric_resolution_sec)
        F1 = results['F1']
        ER = results['ER']
        logs['F1'] = F1
        logs['ER'] = ER

        self.current_F1 = F1

        if self.current_F1 > self.best_F1 + self.considered_improvement:
            self.best_F1 = self.current_F1
            self.model.save_weights(self.file_weights)
            print('F1 = {:.4f}, ER = {:.4f} -  Best val F1: {:.4f} (IMPROVEMENT, saving)\n'.format(
                self.current_F1, ER, self.best_F1))
            self.epochs_since_improvement = 0
            self.epoch_best = epoch
        else:
            print('F1 = {:.4f}, ER = {:.4f} - Best val F1: {:.4f} ({:d})\n'.format(
                self.current_F1, ER, self.best_F1, self.epoch_best))
            self.epochs_since_improvement += 1
        if self.epochs_since_improvement >= self.early_stopping-1:
            print('Not improvement for %d epochs, stopping the training' %
                  self.early_stopping)
            self.model.stop_training = True
