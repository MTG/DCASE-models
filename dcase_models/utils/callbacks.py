from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
eps = 1e-6
from .metrics import accuracy

class MetricsCallback(Callback):
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
        acc,_,_ = accuracy(self.model, self.X_val, self.Y_val)
        logs['Acc'] = acc

        self.current_acc = acc

        if self.current_acc > self.best_acc + self.considered_improvement:
            self.best_acc = self.current_acc
            self.model.save_weights(self.file_weights)
            print('Acc = {:.4f} -  Best val Acc: {:.4f} (IMPROVEMENT, saving)\n'.format(self.current_acc, self.best_acc))
            self.epochs_since_improvement = 0
            self.epoch_best = epoch
        else:
            print('Acc = {:.4f} - Best val Acc: {:.4f} ({:d})\n'.format(self.current_acc, self.best_acc, self.epoch_best))
            self.epochs_since_improvement += 1
        if self.epochs_since_improvement  >= self.early_stopping-1:
            print('Not improvement for %d epochs, stopping the training' % self.early_stopping)
            self.model.stop_training = True