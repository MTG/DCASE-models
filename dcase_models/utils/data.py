import os
# import numpy as np


def get_fold_val(fold_test, fold_list):
    N_folds = len(fold_list)
    fold_test_ix = [k for k, v in enumerate(fold_list) if v == fold_test][0]
    # sum(divmod(fold_test_ix+1,N_folds))
    fold_val_ix = (fold_test_ix+1) % N_folds
    fold_val = fold_list[fold_val_ix]
    return fold_val


def evaluation_setup(fold_test, folds, evaluation_mode,
                     use_validate_set=True):
    if evaluation_mode == 'cross-validation':
        fold_val = get_fold_val(fold_test, folds)
        folds_train = folds.copy()  # list(range(1,N_folds+1))
        folds_train.remove(fold_test)
        if use_validate_set:
            folds_train.remove(fold_val)
            folds_val = [fold_val]
        else:
            folds_val = folds_train.copy()
        folds_test = [fold_test]
    elif evaluation_mode == 'train-validate-test':
        folds_train = ['train']
        folds_val = ['validate']
        folds_test = ['test']
    elif evaluation_mode == 'train_test':
        folds_train = ['train']
        folds_val = ['train']
        folds_test = ['test']
    elif evaluation_mode == 'cross-validation-with-test':
        folds_train = folds.copy()
        fold_val = get_fold_val(fold_test, folds)
        folds_train.remove(fold_val)
        folds_val = [fold_val]
        folds_test = ['test']
    else:
        raise AttributeError("Incorrect evaluation_mode %s" % evaluation_mode)    

    return folds_train, folds_val, folds_test


def check_model_exists(path):
    file_weights = os.path.join(path, "best_weights.hdf5")
    file_json = os.path.join(path, "model.json")
    file_scaler = os.path.join(path, "scaler.pickle")
    return (os.path.exists(file_weights) &
            os.path.exists(file_json) &
            os.path.exists(file_scaler))
