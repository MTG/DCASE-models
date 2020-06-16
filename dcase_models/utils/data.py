import os
# import numpy as np


def get_fold_val(fold_test, fold_list):
    N_folds = len(fold_list)
    fold_test_ix = [k for k, v in enumerate(fold_list) if v == fold_test][0]
    # sum(divmod(fold_test_ix+1,N_folds))
    fold_val_ix = (fold_test_ix+1) % N_folds
    fold_val = fold_list[fold_val_ix]
    return fold_val



def get_data_train_list(folds_data, fold_test, evaluation_mode,
                        scaler=None, upsampling=True):
    fold_list = list(folds_data.keys())

    if evaluation_mode == "cross-validation":
        fold_val = get_fold_val(fold_test, fold_list)
        folds_train = fold_list.copy()  # list(range(1,N_folds+1))
        folds_train.remove(fold_test)
        folds_train.remove(fold_val)

        X_val = folds_data[fold_val]['X']
        Y_val = folds_data[fold_val]['Y']
        if scaler is not None:
            X_val = scaler.transform(X_val)
        X_train = []
        Y_train = []
        for fold in folds_train:
            X_fold = folds_data[fold]['X']
            # print(type(X_fold), len(X_fold))
            if scaler is not None:
                X_fold = scaler.transform(X_fold)
            X_train.extend(X_fold)
            Y_train.extend(folds_data[fold]['Y'])

    return X_train, Y_train, X_val, Y_val

def get_fold_list(folds, evaluation_mode):
    if evaluation_mode == 'cross-validation':
        fold_list = folds
    elif evaluation_mode == 'train_validate_test':
        fold_list = [folds[2]]  # only test fold
    elif evaluation_mode == 'train_test':
        fold_list = [folds[1]]  # only test fold
    else:
        raise AttributeError("incorrect evaluation_mode %s" % evaluation_mode)

    return fold_list


def check_model_exists(path):
    file_weights = os.path.join(path, "best_weights.hdf5")
    file_json = os.path.join(path, "model.json")
    file_scaler = os.path.join(path, "scaler.pickle")
    return (os.path.exists(file_weights) &
            os.path.exists(file_json) &
            os.path.exists(file_scaler))
