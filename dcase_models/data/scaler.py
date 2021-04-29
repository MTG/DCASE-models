from dcase_models.data.data_generator import DataGenerator
import numpy as np
from sklearn.preprocessing import StandardScaler
import inspect


class Scaler():
    """ Scaler object to normalize or scale the data.

    Parameters
    ----------
    normalizer : {'standard' or 'minmax'}, default='standard'
        Type of normalizer.

    Attributes
    ----------
    scaler : sklearn.preprocessing.StandardScaler or list
        Scaler object for standard normalizer or list for minmax scaler.

    See also
    --------
    DataGenerator : data generator class


    Examples
    --------
    >>> from dcase_models.data.scaler import Scaler
    >>> import numpy as np
    >>> scaler = Scaler('minmax')
    >>> X = 3 * np.random.rand(10, 150)
    >>> print(np.amin(X), np.amax(X))

    >>> scaler.fit(X)
    >>> X = scaler.transform(X)
    >>> print(np.amin(X), np.amax(X))

    """
    def __init__(self, normalizer='standard'):
        """ Initialize the Scaler.

        If normalizer is 'standard', initialize the sklearn object.
        """
        self.normalizer = normalizer
        if type(normalizer) is not list:
            self.normalizer = [normalizer]

        self.scaler = []
        for norm in self.normalizer:
            if norm == 'standard':
                self.scaler.append(StandardScaler())
            elif norm == 'minmax':
                self.scaler.append([])
            else:
                self.scaler.append(None)

    def fit(self, X, inputs=True):
        """ Fit the Scaler.

        Parameters
        ----------
        X : ndarray or DataGenerator
            Data to be used in the fitting process.

        """
        if (DataGenerator in inspect.getmro(X.__class__)):
            for batch_index in range(len(X)):
                if inputs:
                    X_batch, _ = X.get_data_batch(batch_index)
                else:
                    _, X_batch = X.get_data_batch(batch_index)
                self.partial_fit(X_batch)
            return True
        else:
            self.partial_fit(X)

    def partial_fit(self, X):
        """ Fit the Scaler in one batch.

        Parameters
        ----------
        X : ndarray
            Data to be used in the fitting process.

        """

        if (len(self.normalizer) == 1) and (type(X) != list):
            X = [X]
        assert type(X) == list
        assert len(self.normalizer) == len(X)

        for j in range(len(self.normalizer)):
            Xj = X[j]
            if type(Xj) == list:
                Xj = np.concatenate(Xj, axis=0)
            if self.normalizer[j] == 'standard':
                X_bands = Xj.shape[-1]
                self.scaler[j].partial_fit(np.reshape(Xj, (-1, X_bands)))
                assert len(self.scaler[j].mean_) == X_bands
            if self.normalizer[j] == 'minmax':
                min_v = np.amin(Xj)
                max_v = np.amax(Xj)
                if len(self.scaler[j]) > 0:
                    min_v = min(min_v, self.scaler[j][0])
                    max_v = max(max_v, self.scaler[j][1])

                self.scaler[j] = [min_v, max_v]

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
        return_list = True
        if (len(self.normalizer) == 1) and (type(X) != list):
            X = [X]
            return_list = False

        for j in range(len(self.normalizer)):
            if type(X[j]) == list:
                for k in range(len(X[j])):
                    X[j][k] = self._apply_transform(X[j][k], j)
            else:
                X[j] = self._apply_transform(X[j], j)

        if not return_list:
            X = X[0]
        return X

    def _apply_transform(self, X, scaler_ix):
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
        if self.normalizer[scaler_ix] == 'standard':
            X_dims = X.shape
            X_bands = X.shape[-1]
            # times = X.shape[0]*X.shape[1]
            X_temp = np.reshape(X, (-1, X_bands))
            X_temp = self.scaler[scaler_ix].transform(X_temp)
            X = X_temp.reshape(X_dims)
        if self.normalizer[scaler_ix] == 'minmax':
            X = 2*((X-self.scaler[scaler_ix][0]) /
                   (self.scaler[scaler_ix][1]-self.scaler[scaler_ix][0])-0.5)
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
        # TODO: How the list self.normalizer should work here.
        scaler_ix = 0
        if self.normalizer[scaler_ix] == 'minmax':
            X = (self.scaler[scaler_ix][1]-self.scaler[scaler_ix][0]) * \
                (X/2. + 0.5) + self.scaler[scaler_ix][0]
        if self.normalizer[scaler_ix][0] == 'standard':
            X = self.scaler.inverse_transform(X)
        return X
