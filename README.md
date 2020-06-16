# DCASE-models
DCASE-models is a python library that aims to be a general structure to define, train and evaluate models for DCASE related problems. The idea is to have a modular and easy-to-use library to rapid prototype experiments. Regarding this, we create a class based structure for the differente stages related to a classifier of audio signals: feature extractor, data generator, scaler and the model itself. The library uses librosa for feature extraction and keras for the classifier, but given the modularity, the user can uses other platforms.

## Organization of the repository 

````
root/
|
|- dcase_models/ ______________________ # Library source files
|  |- data/ ___________________________ # Data related source files
|  |- model/ __________________________ # Models related source files
|  |- util/ ___________________________ # Utils
|
|- tests/ _____________________________ # Scripts for unit tests
|
````

## Installation instructions
We recommend to install DCASE-models in a dedicated virtual environment. For instance, using anaconda:
```
conda create -n dcase python=3.6
conda activate dcase
```
For GPU support:
```
conda install cudatoolkit cudnn
```
Then to install the package:
```
git clone https://github.com/pzinemanas/DCASE-models.git
cd DCASE-models
pip install .
```
To include visualization related dependencies, run:
```
pip install .[visualization]
```

## Usage
There are several ways to use this library.

### Python script
The folder `scripts` includes python scripts for data downloading, feature extraction, model training and testing. This examples show how to use DCASE-models within a python script.

### Jupyter notebooks
The folder `notebooks` includes a list of notebooks that replicate experiments using DCASE-models.

### Web applications
The folder `visualization` includes a user interface to define, train and visualize the models defined in this library.

Go to DCASE-models folder and run:
```
python -m visualization.index
```


## Expand DCASE-models
This library is devised to be easy to expand in order to be fitted to your specific application. You can defined custom feature extractors, datasets, models, etc. 

For instance to define a new extractor that calculates chroma features using librosa:
```
class Chroma(FeatureExtractor):
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=512, n_fft=1024, sr=44100,
                 # Add here your custom parameters
                 n_chroma=12):
                 
        # Don't forget this line
        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         n_fft=n_fft, sr=sr)

        # Add your custom parameters to self.params
        self.params['name'] = 'Chroma'
        self.params['n_chroma'] = n_chroma

    def calculate_features(self, file_name):
        # Here define your function to calculate the chroma features

        # Load audio
        audio = self.load_audio(file_name)
        
        # Get chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr, 
                                             n_fft=self.n_fft,
                                             hop_length=audio_hop,
                                             win_length=audio_win)

        # convert to sequences (windowing)
        chroma_seqs = self.get_sequences(chroma, pad=True)

        # convert to numpy
        chroma_np = np.asarray(chroma_seqs)

        # transpose time and freq dims
        chroma_np = np.transpose(chroma_np, (0, 2, 1))

        return chroma_np
```

