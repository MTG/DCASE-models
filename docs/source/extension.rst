Extending the library
=====================

This section includes clear instructions on how to extend different components of `DCASE-models`.

Datasets
--------

Each dataset is implemented in the library as a class that inherits from :class:`~dcase_models.data.Dataset`.

To include a new dataset in the library you should extend the :class:`~dcase_models.data.Dataset` class and implement:

 - :code:`__init__` , where you can define and store arguments related to the dataset.
 - :code:`build` , where you define the fold list, label list, paths, etc.
 - :code:`generate_file_lists` , where you define the dataset structure.
 - :code:`get_annotations` , where you implement the function to get the annotations from a given audio file.
 - :code:`download` , where you implement the steps to download and decompress the dataset.

Below we follow all the necessary steps to implement a new dataset.
Let's assume that the new dataset has two labels (dog and cat), three folds (train, validate and test), and the audio files are stored in :code:`DATASET_PATH/audio`.
Besides the new dataset has the following structure::

    DATASET_PATH/
    |
    |- audio/
    |  |- train
    |  |  |- file1-0-X.wav
    |  |  |- file2-1-X.wav
    |  |  |- file3-0-X.wav
    |  |
    |  |- validate
    |  |  |- file1-1-Y.wav
    |  |  |- file2-0-Y.wav
    |  |  |- file3-1-Y.wav
    |  |
    |  |- test
    |  |  |- file1-1-Z.wav
    |  |  |- file2-0-Z.wav
    |  |  |- file3-0-Z.wav


Note that each fold has a folder inside the audio path. Also the file name includes the class label coded after 
the first dash character (0 for dog, 1 for cat).

The first step is to create a new class that inherits from :class:`~dcase_models.data.Dataset`,
and implement its :code:`__init__()` method.
Since the only argument needed for this custom dataset is its path, we simply initialize the :code:`super().__init__()` method.
If your dataset needs other arguments from the user, add them here.

.. code-block:: python

    from dcase_models.data.dataset_base import Dataset


    class CustomDataset(Dataset):
        def __init__(self, dataset_path):
            # Don't forget to add this line
            super().__init__(dataset_path)

Now implement the :meth:`~dcase_models.data.Dataset.build` method.
You should define here the :code:`audio_path`, :code:`fold_list` and :code:`label_list` attributes.
You can also define other attributes for your dataset.

.. code-block:: python

        def build(self):
            self.audio_path = os.path.join(self.dataset_path, 'audio')
            self.fold_list = ["train", "validate", "test"]
            self.label_list = ["dog", "cat"]
            self.evaluation_mode = 'train-validate-test'

The :meth:`~dcase_models.data.Dataset.generate_file_lists` method defines the structure of the dataset. 
Basically this structure is defined in the :code:`self.file_lists` dictionary. 
This dictionary stores the list of the paths to the audio files for each fold in the dataset. 
Note that you can use the :func:`~dcase_models.util.list_wav_files` function to list all wav files in a given path.

.. code-block:: python

        def generate_file_lists(self):
            for fold in self.fold_list:
                audio_folder = os.path.join(self.audio_path, fold)
                self.file_lists[fold] = list_wav_files(audio_folder)

Now let's define :meth:`~dcase_models.data.Dataset.get_annotations`.
This method receives three arguments: the path to the audio file, the features representation and the time resolution 
(used when the annotations are defined following a fix time-grid, e.g see :class:`~dcase_models.data.URBAN_SED`).
Note that the first dimension (index sequence) of the annotations and the feature representation coincide.
In this example the label of each audio file is coded in its name as explained before. 

.. code-block:: python

        def get_annotations(self, file_name, features, time_resolution):
            y = np.zeros((len(features), len(self.label_list)))
            class_ix = int(os.path.basename(file_name).split('-')[1])
            y[:, class_ix] = 1
            return y

The :meth:`~dcase_models.data.Dataset.download` method defines the steps for downloading the dataset.
You can use the :meth:`~dcase_models.data.Dataset.download` method from the parent :class:`~dcase_models.data.Dataset` 
to download and decompress all files from zenodo.
Also you can use :func:`~dcase_models.util.move_all_files_to_parent` function to move all files from a subdirectory to the parent.

.. code-block:: python

        def download(self, force_download=False):
            zenodo_url = "https://zenodo.org/record/1234567/files"
            zenodo_files = ["CustomDataset.tar.gz"]
            downloaded = super().download(
                zenodo_url, zenodo_files, force_download
            )
            if downloaded:
                # mv self.dataset_path/CustomDataset/* self.dataset_path/
                move_all_files_to_parent(self.dataset_path, "CustomDataset")
                # Don't forget this line
                self.set_as_downloaded()

.. note::
    If you implement a class for a publicly available dataset that is not present in :class:`~dcase_models.data.Dataset`, 
    consider filing a Github issue or, even better, sending us a pull request.


Features
--------

Feature representations are implemented as specializations of the base class :class:`~dcase_models.data.FeatureExtractor`.

In order to implement a new feature you should write a class that inherits from :class:`~dcase_models.data.FeatureExtractor`.

The methods you should reimplement are: 

 - :code:`__init__` , where you can define and store the features arguments.
 - :code:`calculate` , where you define the feature calculation process.

For instance, if you want to implement Chroma features:

.. code-block:: python

    import numpy as np
    import librosa
    from dcase_models.data.features import FeatureExtractor


    class Chroma(FeatureExtractor):
        def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                    audio_win=1024, audio_hop=680, sr=22050,
                    n_fft=1024, n_chroma=12, pad_mode='reflect'):

            super().__init__(sequence_time=sequence_time,
                            sequence_hop_time=sequence_hop_time,
                            audio_win=audio_win, audio_hop=audio_hop,
                            sr=sr)

            self.n_fft = n_fft
            self.n_chroma = n_chroma
            self.pad_mode = pad_mode

        def calculate(self, file_name):
            # Load the audio signal
            audio = self.load_audio(file_name)

            # Pad audio signal
            if self.pad_mode is not None:
                audio = librosa.util.fix_length(
                    audio,
                    audio.shape[0] + librosa.core.frames_to_samples(
                        self.sequence_frames, self.audio_hop, n_fft=self.n_fft),
                    axis=0, mode=self.pad_mode
                )

            # Get the spectrogram, shape (n_freqs, n_frames)
            stft = librosa.core.stft(audio, n_fft=self.n_fft,
                                    hop_length=self.audio_hop,
                                    win_length=self.audio_win, center=False)
            # Convert to power
            spectrogram = np.abs(stft)**2

            # Convert to chroma_stft, shape (n_chroma, n_frames)
            chroma = librosa.feature.chroma_stft(
                S=spectrogram, sr=self.sr, n_fft=self.n_fft, n_chroma=self.n_chroma)

            # Transpose time and freq dims, shape (n_frames, n_chroma)
            chroma = chroma.T

            # Windowing, creates sequences
            chroma = np.ascontiguousarray(chroma)
            chroma = librosa.util.frame(
                chroma, self.sequence_frames, self.sequence_hop, axis=0
            )

            return chroma

Models
------

The models are implemented as specializations of the base class :class:`~dcase_models.models.KerasModelContainer`.

To include a new model in the library you should extend the :class:`~dcase_models.models.KerasModelContainer` class and implement the following methods:

 - :code:`__init__` , where you can define and store the model arguments.
 - :code:`build` , where you define the model architecture.

Note that you might also reimplement the :meth:`~dcase_models.model.KerasModelContainer.train` method.
This specially useful for complex models (multiple inputs and outputs, custom loss functions, etc.)

For instance, to implement a simple Convolutional Neural Network:

.. code-block:: python

    from keras.layers import Input, Lambda, Conv2D, MaxPooling2D
    from keras.layers import Dropout, Dense, Flatten
    from keras.layers import BatchNormalization
    from keras.models import Model
    import keras.backend as K
    from dcase_models.model.container import KerasModelContainer


    class CNN(KerasModelContainer):
        def __init__(self, model=None, model_path=None,
                    metrics=['classification'], n_classes=10,
                    n_frames=64, n_freqs=128):

            self.n_classes = n_classes
            self.n_frames = n_frames
            self.n_freqs = n_freqs

            # Don't forget this line
            super().__init__(model=model, model_path=model_path,
                            model_name='MLP', metrics=metrics)

        def build(self):
            # input
            x = Input(shape=(self.n_frames, self.n_freqs), dtype='float32', name='input')

            # expand dims
            y = Lambda(lambda x: K.expand_dims(x, -1), name='expand_dims')(x)

            # CONV 1
            y = Conv2D(24, (5, 5), padding='valid',
                       activation='relu', name='conv1')(y)
            y = MaxPooling2D(pool_size=(2, 2), strides=None,
                             padding='valid', name='maxpool1')(y)
            y = BatchNormalization(name='batchnorm1')(y)

            # CONV 2
            y = Conv2D(24, (5, 5), padding='valid',
                       activation='relu', name='conv2')(y)
            y = BatchNormalization(name='batchnorm2')(y)

            # Flatten and Dropout
            y = Flatten(name='flatten')(y)
            y = Dropout(0.5, name='dropout1')(y)

            # Dense layer
            y = Dense(self.n_classes, activation='softmax', name='out')(y)

            # Create model
            self.model = Model(inputs=x, outputs=y, name='model')

            # Don't forget this line
            super().build()