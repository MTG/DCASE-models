from .data_generator import DataGenerator
import numpy as np
from sklearn.preprocessing import StandardScaler
import inspect


class Scaler():
    def __init__(self, normalizer='standard'):
        self.normalizer = normalizer
        if normalizer == 'standard':
            self.scaler = StandardScaler()
        if normalizer == 'minmax':
            self.scaler = []

    def fit(self, mel):
        if (DataGenerator in inspect.getmro(mel.__class__)):
            for batch_index in range(len(mel)):
                X, Y = mel.get_data_batch(batch_index)
                self.partial_fit(X)
            return True

        print('NO!')
        if type(mel) == list:
            mel = np.concatenate(mel, axis=0)

        if self.normalizer == 'standard':
            mel_bands = mel.shape[-1]
            self.scaler.fit(np.reshape(mel, (-1, mel_bands)))
            assert len(self.scaler.mean_) == mel_bands
        if self.normalizer == 'minmax':
            min_v = np.amin(mel)  # ,axis=(0,2))
            max_v = np.amax(mel)  # ,axis=(0,2))
            self.scaler = [min_v, max_v]

    def partial_fit(self, mel):
        if type(mel) == list:
            mel = np.concatenate(mel, axis=0)
        if self.normalizer == 'standard':
            mel_bands = mel.shape[-1]
            self.scaler.partial_fit(np.reshape(mel, (-1, mel_bands)))
            assert len(self.scaler.mean_) == mel_bands
        if self.normalizer == 'minmax':
            min_v = np.amin(mel)
            max_v = np.amax(mel)
            if len(self.scaler) > 0:
                min_v = min(min_v, self.scaler[0])
                max_v = max(max_v, self.scaler[1])
  
            self.scaler = [min_v, max_v]

    def transform(self, mel):
        if type(mel) == list:
            for j in range(len(mel)):
                mel[j] = self.apply_transform(mel[j])
        else:
            mel = self.apply_transform(mel)
        return mel

    def apply_transform(self, mel):
        if self.normalizer == 'standard':
            mel_dims = mel.shape
            mel_bands = mel.shape[-1]
            # times = mel.shape[0]*mel.shape[1]
            mel_temp = np.reshape(mel, (-1, mel_bands))
            mel_temp = self.scaler.transform(mel_temp)
            mel = mel_temp.reshape(mel_dims)
        if self.normalizer == 'minmax':
            mel = 2*((mel-self.scaler[0]) /
                        (self.scaler[1]-self.scaler[0])-0.5)
        return mel

    def inverse_transform(self, mel):
        if self.normalizer == 'minmax':
            mel = (self.scaler[1]-self.scaler[0]) * \
                (mel/2. + 0.5) + self.scaler[0]
        if self.normalizer == 'standard':
            mel = self.scaler.inverse_transform(mel)
        return mel
