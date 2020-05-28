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


def load_training_log(weights_folder,fold_test,row_ix=10):
    log_path = os.path.join(weights_folder, 'fold' + str(fold_test) + '_training.log')
    epochs = []
    val_accs = []
    if os.path.exists(log_path):
        df = read_csv(log_path)
        epochs = list(df['epoch'].values)

        for column in ['val_logits_acc','val_out_acc','val_acc']:
            if column in df.columns:
                val_accs = list(df[column].values)
                break
    return epochs,val_accs

import wget
import zipfile
import shutil
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
        print(www_path)
        wget.download(www_path, dataset_folder)
        print()
    print('Done!')

    print('Audio files extraction ... ')
    # extract each zip file
    for zip_file in zenodo_files:
        zip_file_path = os.path.join(dataset_folder, zip_file)
        if not os.path.exists(zip_file_path):
            all_files = [f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))]
            for f in all_files:
                if f.split('-')[-1] == zip_file:
                    zip_file_path = os.path.join(dataset_folder, f)
        print('Extracting file: ', zip_file_path)
        #zip_ref = zipfile.ZipFile(zip_file_path) # create zipfile object
        #zip_ref.extractall(dataset_folder) # extract file to dir
        #zip_ref.close() # close file
        
        shutil.unpack_archive(zip_file_path, dataset_folder)
        os.remove(zip_file_path) # delete zipped file
    print('Done!')


def move_all_files_to_parent(parent, child):
    source = os.path.join(parent, child)
    files = os.listdir(source)
    for f in files:
        shutil.move(os.path.join(source, f), os.path.join(parent, f))
    shutil.rmtree(source)