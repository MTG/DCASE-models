import wget
import csv
import shutil
import pickle
import os
import json


def save_pickle(X, path):
    with open(path, 'wb') as f:
        pickle.dump(X, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        X = pickle.load(f)
    return X


def save_json(path, json_string):
    with open(path, 'w') as outfile:
        json.dump(json_string, outfile)


def load_json(path):
    with open(path) as json_f:
        data = json.load(json_f)
    return data


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def duplicate_folder_structure(origin_path, destination_path):
    for dirpath, dirnames, filenames in os.walk(origin_path):
        structure = os.path.join(destination_path, dirpath[len(origin_path)+1:])
        mkdir_if_not_exists(structure)

def list_wav_files_in_folder(path):
    wav_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for file_audio in filenames:
            path_to_file_audio = os.path.join(dirpath, file_audio)
            if path_to_file_audio.endswith('wav'):
                wav_files.append(path_to_file_audio)
    return wav_files



def load_training_log(weights_folder):
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
    # adapted from Rocamora's code
    # https://gitlab.fing.edu.uy/urban-noise-monitoring/alarm-monitoring/-/blob/master/dataset/get_MAVD_audio.py

    mkdir_if_not_exists(dataset_folder)

    print('Audio files download ... ')
    for zip_file in zenodo_files:
        zip_file_path = os.path.join(dataset_folder, zip_file)
        if os.path.exists(zip_file_path):
            print('File %s exists, skipping...' % zip_file)
            continue
        print('Downloading file: ', zip_file)
        www_path = os.path.join(zenodo_url, zip_file)
        wget.download(www_path, dataset_folder)
        print()
    print('Done!')

    print('Audio files extraction ... ')
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
    source = os.path.join(parent, child)
    files = os.listdir(source)
    for f in files:
        shutil.move(os.path.join(source, f), os.path.join(parent, f))
    shutil.rmtree(source)
