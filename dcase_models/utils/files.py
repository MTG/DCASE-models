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