# encoding: utf-8
"""Data functions"""

def get_fold_val(fold_test, fold_list):
    """ Get the validation fold given the test fold.

    Useful for cross-validation evaluation mode.

    Return the next fold in a circular way.
    e.g. if the fold_list is ['fold1', 'fold2',...,'fold10'] and
    the fold_test is 'fold1', then return 'fold2'.
    If the fold_test is 'fold10', return 'fold1'.

    Parameters
    ----------
    fold_test : str
        Fold used for model testing.
    fold_list : list of str
        Fold list.

    Returns
    -------
    str
        Validation fold.

    """
    N_folds = len(fold_list)
    fold_test_ix = [k for k, v in enumerate(fold_list) if v == fold_test][0]
    # sum(divmod(fold_test_ix+1,N_folds))
    fold_val_ix = (fold_test_ix+1) % N_folds
    fold_val = fold_list[fold_val_ix]
    return fold_val


def evaluation_setup(fold_test, folds, evaluation_mode,
                     use_validate_set=True):
    """ Return a evaluation setup given by the evaluation_mode.

    Return fold list for training, validatin and testing the model.

    Each evaluation_mode return different lists.

    Parameters
    ----------
    fold_test : str
        Fold used for model testing.
    folds : list of str
        Fold list.
    evaluation_mode : str
        Evaluation mode ('cross-validation', 'train-validate-test',
        'cross-validation-with-test', 'train-test')
    use_validate_set : bool
        If not, the validation set is the same as the train set.

    Returns
    -------
    list
        List of folds for training
    list
        List of folds for validating
    list
        List of folds for testing

    """
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
    elif evaluation_mode == 'train-test':
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
