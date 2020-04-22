import sys
import os
import numpy as np
from librosa.core import power_to_db
import matplotlib.pyplot as plt

sys.path.append('../')
# only for vscode
parent_path = os.path.abspath(os.path.join(sys.path[0], os.pardir))
sys.path.append(parent_path)

from apnet.data.feature_extractor import FeatureExtractor

params = {'sequence_time': 1.0, 
          'sequence_hop_time':0.5,
          'audio_hop':512, 'audio_win':1024,
          'n_fft':1024, 'sr': 22050, 'mel_bands': 128,
          'features': ['melspec'], 'augmentation': {'pitch_shift': -2} } #{'time_stretch': {'rate': 0.9}}


# extract features and save files
feature_extractor = FeatureExtractor(**params)
feature_extractor.extract('audio/','features/')

# load features files and show them
mel_spec = np.load('features/mel_spectrograms/40722-8-0-7.npy')
spec = np.load('features/spectrograms/40722-8-0-7.npy')

n_sequences = mel_spec.shape[0]
#mel_spec = power_to_db(mel_spec)
#spec = power_to_db(spec)
print(mel_spec.shape, spec.shape, n_sequences)

plt.figure()
for j in range(n_sequences):
    plt.subplot(1, n_sequences, j+1)
    plt.imshow(mel_spec[j].T)
plt.show()

feature_extractor.save_mel_basis('features/mel_basis.npy')
feature_extractor.save_parameters_json('features/parameters.json')