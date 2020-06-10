import numpy as np
from sklearn.preprocessing import StandardScaler


class Scaler():
    def __init__(self, normalizer='standard', bands=None):
        self.normalizer = normalizer
        self.bands = bands
        if normalizer == 'standard':
            self.scaler = StandardScaler()
        if normalizer == 'minmax':
            self.scaler = []

    def fit(self, mel):
        if type(mel) == list:
            mel = np.concatenate(mel, axis=0)

        if self.normalizer == 'standard':
            mel_bands = mel.shape[-1]
            self.scaler.fit(np.reshape(mel, (-1, mel_bands)))
            assert len(self.scaler.mean_) == mel_bands
        if self.normalizer == 'minmax':
            if self.bands is None:
                min_v = np.amin(mel)  # ,axis=(0,2))
                max_v = np.amax(mel)  # ,axis=(0,2))
                self.scaler = [min_v, max_v]
            else:
                for i in range(1, len(self.bands)):
                    min_band = self.bands[i-1]
                    max_band = self.bands[i]
                    print(min_band, max_band)
                    mel_i = mel[:, :, min_band:max_band]

                    min_v = np.amin(mel_i)
                    max_v = np.amax(mel_i)

                    self.scaler.append([min_v, max_v])
                    print(i, mel_i.shape)

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
            times = mel.shape[0]*mel.shape[1]
            mel_temp = mel.reshape(times, mel_bands)
            mel_temp = self.scaler.transform(mel_temp)
            mel = mel_temp.reshape(mel_dims)
        if self.normalizer == 'minmax':
            if self.bands is None:
                mel = 2*((mel-self.scaler[0]) /
                         (self.scaler[1]-self.scaler[0])-0.5)
            else:
                for i in range(1, len(self.bands)):
                    min_band = self.bands[i-1]
                    max_band = self.bands[i]
                    scaler = self.scaler[i-1]
                    mel[:, :, min_band:max_band] = 2 * \
                        ((mel[:, :, min_band:max_band]-scaler[0]) /
                         (scaler[1]-scaler[0])-0.5)
        return mel

    def antitransform(self, mel):
        if self.normalizer == 'minmax':
            mel = (self.scaler[1]-self.scaler[0]) * \
                (mel/2. + 0.5) + self.scaler[0]

        return mel
