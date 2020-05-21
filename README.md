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

## Usage
There is two different ways for using this library:
1) Clone the library and change/overwrite whatever you want
2) Import this library and re-define some classes.
