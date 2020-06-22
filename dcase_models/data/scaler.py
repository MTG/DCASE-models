from .data_generator import DataGenerator
import numpy as np
from sklearn.preprocessing import StandardScaler
import inspect


class Scaler():
    """ Scaler object to normalize or scale the data.

    Attributes
    ----------
    scaler : sklearn.preprocessing.StandardScaler or list
        Scaler object for standard normalizer or list for
        minmax scaler.

    Methods
    -------
    fit(X):
        Fit the scaler.
    partial_fit(X)
        Fit the scaler in one batch.
    transform(X):
        Scale X using the scaler.
    inverse_transform(X):
        Inverse scale.

    """
    def __init__(self, normalizer='standard'):
        """ Initialize the Scaler.

        If normalizer is 'standard', initialize the sklearn object.

        Parameters
        ----------
        normalizer : str
            Type of normalizer ('standard' or 'minmax')

        """
        self.normalizer = normalizer
        if normalizer == 'standard':
            self.scaler = StandardScaler()
        if normalizer == 'minmax':
            self.scaler = []

    def fit(self, X):
        """ Fit the Scaler.

        Parameters
        ----------
        X : ndarray or DataGenerator
            Data to be used in the fitting process.

        """
        if (DataGenerator in inspect.getmro(X.__class__)):
            for batch_index in range(len(X)):
                X_batch, _ = X.get_data_batch(batch_index)
                self.partial_fit(X_batch)
            return True

        if type(X) == list:
            X = np.concatenate(X, axis=0)

        if self.normalizer == 'standard':
            X_bands = X.shape[-1]
            self.scaler.fit(np.reshape(X, (-1, X_bands)))
            assert len(self.scaler.mean_) == X_bands
        if self.normalizer == 'minmax':
            min_v = np.amin(X)  # ,axis=(0,2))
            max_v = np.amax(X)  # ,axis=(0,2))
            self.scaler = [min_v, max_v]

    def partial_fit(self, X):
        """ Fit the Scaler in one batch.

        Parameters
        ----------
        X : ndarray
            Data to be used in the fitting process.

        """
        if type(X) == list:
            X = np.concatenate(X, axis=0)
        if self.normalizer == 'standard':
            X_bands = X.shape[-1]
            self.scaler.partial_fit(np.reshape(X, (-1, X_bands)))
            assert len(self.scaler.mean_) == X_bands
        if self.normalizer == 'minmax':
            min_v = np.amin(X)
            max_v = np.amax(X)
            if len(self.scaler) > 0:
                min_v = min(min_v, self.scaler[0])
                max_v = max(max_v, self.scaler[1])
  
            self.scaler = [min_v, max_v]

    def transform(self, X):
        """ Scale X using the scaler.

        Parameters
        ----------
        X : ndarray
            Data to be scaled.

        Returns
        -------
        ndarray
            Scaled data. The shape of the output is the same
            of the input.

        """
        if type(X) == list:
            for j in range(len(X)):
                X[j] = self._apply_transform(X[j])
        else:
            X = self._apply_transform(X)
        return X

    def _apply_transform(self, X):
        """ Helper of transform()

        Parameters
        ----------
        X : ndarray
            Data to be scaled.

        Returns
        -------
        ndarray
            Scaled data. The shape of the output is the same
            of the input.

        """
        if self.normalizer == 'standard':
            X_dims = X.shape
            X_bands = X.shape[-1]
            # times = X.shape[0]*X.shape[1]
            X_temp = np.reshape(X, (-1, X_bands))
            X_temp = self.scaler.transform(X_temp)
            X = X_temp.reshape(X_dims)
        if self.normalizer == 'minmax':
            X = 2*((X-self.scaler[0]) /
                (self.scaler[1]-self.scaler[0])-0.5)
        return X

    def inverse_transform(self, X):
        """ Invert transformation.

        Parameters
        ----------
        X : ndarray
            Data to be scaled.

        Returns
        -------
        ndarray
            Scaled data. The shape of the output is the same
            of the input.

        """
        if self.normalizer == 'minmax':
            X = (self.scaler[1]-self.scaler[0]) * \
                (X/2. + 0.5) + self.scaler[0]
        if self.normalizer == 'standard':
            X = self.scaler.inverse_transform(X)
        return X
