# encoding: utf-8
"""Events functions"""

import wget
import csv
import shutil
import pickle
import os
import json
import inspect

def save_pickle(X, path):
    """ Save a pickle object in the location given by path.

    Parameters
    ----------
    X : pickle object
        Object to be saved.
    path : str
        Path to pickle file.

    """
    with open(path, 'wb') as f:
        pickle.dump(X, f)


def load_pickle(path):
    """ Load a pickle object from path.

    Parameters
    ----------
    path : str
        Path to pickle file.

    Returns
    -------
    pickle object
        Loaded pickle object.

    """
    with open(path, 'rb') as f:
        X = pickle.load(f)
    return X


def save_json(path, json_string):
    """ Save a json file in the location given by path.

    Parameters
    ----------
    path : str
        Path to json file.
    json_string : str
        JSON string to be saved.

    """
    with open(path, 'w') as outfile:
        json.dump(json_string, outfile)


def load_json(path):
    """ Load a json file from path.

    Parameters
    ----------
    path : str
        Path to json file.

    Returns
    -------
    dict
        Data from the json file.

    """
    with open(path) as json_f:
        data = json.load(json_f)
    return data


def mkdir_if_not_exists(path, parents=False):
    """ Make dir if does not exists.

    If parents is True, also creates all parents needed.

    Parameters
    ----------
    path : str
        Path to folder to be created.
    parents : bool, optional
        If True, also creates all parents needed.

    """
    if not os.path.exists(path):
        if parents:
            os.makedirs(path)
        else:
            os.mkdir(path)


def duplicate_folder_structure(origin_path, destination_path):
    """ Duplicate the folder structure from the origin to the destination.

    Parameters
    ----------
    origin_path : str
        Origin path.
    destination_path : str
        Destination path.

    """
    for dirpath, dirnames, filenames in os.walk(origin_path):
        structure = os.path.join(
            destination_path, dirpath[len(origin_path)+1:]
        )
        try:
            mkdir_if_not_exists(structure)
        except:
            parent_structure = os.path.abspath(
                os.path.join(structure, os.pardir)
            )
            mkdir_if_not_exists(parent_structure)
            mkdir_if_not_exists(structure)


def list_wav_files(path):
    """ List all wav files in the path including subfolders.

    Parameters
    ----------
    path : str
        Path to wav files.

    Returns
    -------
    list
        List of paths to the wav files.

    """
    wav_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for file_audio in filenames:
            path_to_file_audio = os.path.join(dirpath, file_audio)
            if ((file_audio.endswith('wav')) and
               (not file_audio.startswith('.'))):
                wav_files.append(path_to_file_audio)
    return wav_files


def list_all_files(path):
    """ List all files in the path including subfolders.

    Parameters
    ----------
    path : str
        Path to files.

    Returns
    -------
    list
        List of paths to the files.

    """
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for file_ in filenames:
            path_to_file = os.path.join(dirpath, file_)
            files.append(path_to_file)
    return files


def load_training_log(weights_folder):
    """ Load the training log files of keras.

    Parameters
    ----------
    weights_folder : str
        Path to training log folder.

    Returns
    -------
    dict
        Dict with the log information. Each key in the dict
        includes information of some variable.
        e.g. {'loss': [0.1, ...], 'accuracy': [80.1, ...]}

    """
    log_file = 'training.log'
    log_path = os.path.join(weights_folder, log_file)

    log = {}
    if os.path.exists(log_path):
        with open(log_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    measures = row
                    for measure in measures:
                        log[measure] = []
                    line_count += 1
                    continue
                for ix, value in enumerate(row):
                    measure = measures[ix]
                    log[measure].append(value)
        return log
    else:
        return None


def download_files_and_unzip(dataset_folder, zenodo_url, zenodo_files):
    """ Download files from zenodo and decompress them.

    Parameters
    ----------
    dataset_folder : str
        Path to the folder where download the files.
    zenodo_url : str
        Url to the zenodo repository.
    zenodo_files : list of str
        List of file names to download.

    """
    # adapted from Rocamora's code
    # https://gitlab.fing.edu.uy/urban-noise-monitoring/alarm-monitoring/-/blob/master/dataset/get_MAVD_audio.py

    mkdir_if_not_exists(dataset_folder)

    for zip_file in zenodo_files:
        zip_file_path = os.path.join(dataset_folder, zip_file)
        if os.path.exists(zip_file_path):
            print('File %s exists, skipping...' % zip_file)
            continue
        print('Downloading file: ', zip_file)
        www_path = os.path.join(zenodo_url, zip_file)
        wget.download(www_path, dataset_folder)
    print('Done!')

    # extract each zip file
    for zip_file in zenodo_files:
        zip_file_path = os.path.join(dataset_folder, zip_file)
        if not os.path.exists(zip_file_path):
            all_files = [
                f for f in os.listdir(dataset_folder) if
                os.path.isfile(os.path.join(dataset_folder, f))
            ]
            for f in all_files:
                if f.split('-')[-1] == zip_file:
                    zip_file_path = os.path.join(dataset_folder, f)
        print('Extracting file: ', zip_file_path)
        # zip_ref = zipfile.ZipFile(zip_file_path) # create zipfile object
        # zip_ref.extractall(dataset_folder) # extract file to dir
        # zip_ref.close() # close file
        try:
            shutil.unpack_archive(zip_file_path, dataset_folder)
        except:
            continue
        os.remove(zip_file_path)  # delete zipped file
    print('Done!')


def move_all_files_to_parent(parent, child):
    """ Move all files in parent/child to the parent/

    Parameters
    ----------
    parent : str
        Path to the parent folder.
    child : str
        Folder name of the child folder.

    """
    source = os.path.join(parent, child)
    files = os.listdir(source)
    for f in files:
        shutil.move(os.path.join(source, f), os.path.join(parent, f))
    shutil.rmtree(source)


def move_all_files_to(source, destination):
    """ Move all files from source to destination

    Parameters
    ----------
    source : str
        Path to the source folder.
    destination : str
        Folder to the destination folder.

    """
    files = os.listdir(source)
    for f in files:
        shutil.move(os.path.join(source, f), os.path.join(destination, f))
    shutil.rmtree(source)


def example_audio_file(index=0):
    """ Get path to an example audio file

    Parameters
    ----------
    index : int, default=0
        Index of the audio file

    Returns
    -------
    path : str
        Path to the example audio file

    """
    data_path = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(data_path, os.pardir))
    data_path = os.path.join(data_path, 'example_dataset/audio')
    wav_files = list_wav_files(data_path)
    return wav_files[index]
