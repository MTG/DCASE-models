import os
# import numpy as np


def get_fold_val(fold_test, fold_list):
    N_folds = len(fold_list)
    fold_test_ix = [k for k, v in enumerate(fold_list) if v == fold_test][0]
    # sum(divmod(fold_test_ix+1,N_folds))
    fold_val_ix = (fold_test_ix+1) % N_folds
    fold_val = fold_list[fold_val_ix]
    return fold_val


# def get_data_train(folds_data, fold_test, evaluation_mode,
#                    scaler=None, upsampling=True):

#     fold_list = list(folds_data.keys())

#     # Concate all files in the list
#     folds_data_numpy = {}
#     if type(folds_data[fold_list[0]]['Y']) == list:
#         for fold in fold_list:
#             folds_data_numpy[fold] = {}
#             X = []
#             Y = []
#             print(len(folds_data[fold]['Y']), folds_data[fold]['Y'][0].shape)
#             for j in range(len(folds_data[fold]['X'])):
#                 y = np.expand_dims(folds_data[fold]['Y'][j], 0)
#                 features = folds_data[fold]['X'][j]
#                 y = np.repeat(y, len(features), 0)
#                 Y.append(y)
#                 X.append(features)
#             folds_data_numpy[fold]['X'] = np.concatenate(X, axis=0)
#             folds_data_numpy[fold]['Y'] = np.concatenate(Y, axis=0)
#         print(folds_data_numpy[fold]['X'].shape,
#               folds_data_numpy[fold]['Y'].shape)
#     else:
#         folds_data_numpy = folds_data
#     if evaluation_mode == "train_validate_test":
#         print('please check: train, validate, test --> ', fold_list)
#         X_train = folds_data_numpy[fold_list[0]]['X']
#         Y_train = folds_data_numpy[fold_list[0]]['Y']
#         X_val = folds_data_numpy[fold_list[1]]['X']
#         Y_val = folds_data_numpy[fold_list[1]]['Y']
#         X_test = folds_data_numpy[fold_list[2]]['X']
#         Y_test = folds_data_numpy[fold_list[2]]['Y']

#     elif evaluation_mode == "train_test":
#         print('please check: train, test --> ', fold_list)
#         X_train = folds_data_numpy[fold_list[0]]['X']
#         Y_train = folds_data_numpy[fold_list[0]]['Y']
#         X_test = folds_data_numpy[fold_list[1]]['X']
#         Y_test = folds_data_numpy[fold_list[1]]['Y']
#         X_val = 0
#         Y_val = 0

#     elif evaluation_mode == "cross-validation":
#         X_test = folds_data_numpy[fold_test]['X']
#         Y_test = folds_data_numpy[fold_test]['Y']

#         N_folds = len(folds_data)
#         if N_folds > 2:
#             fold_val = get_fold_val(fold_test, fold_list)
#             print('fold_list', fold_list)
#             print('fold_test', fold_test)
#             print('fold_val', fold_val)
#             X_val = folds_data[fold_val]['X']
#             Y_val = folds_data[fold_val]['Y']

#             folds_train = fold_list.copy()  # list(range(1,N_folds+1))
#             folds_train.remove(fold_test)
#             folds_train.remove(fold_val)
#             print('folds_train', folds_train)

#             #print('\nFold' + str(fold_test), 'Fold_val: ' + str(fold_val))
#             X_train = []
#             Y_train = []
#             for fold_train in folds_train:
#                 X_train.append(folds_data_numpy[fold_train]['X'])
#                 Y_train.append(folds_data_numpy[fold_train]['Y'])
#                 print(folds_data_numpy[fold_train]['X'].shape,
#                       folds_data_numpy[fold_train]['Y'].shape)
#             X_train = np.concatenate(X_train, axis=0)
#             Y_train = np.concatenate(Y_train, axis=0)
#         else:
#             X_train = folds_data_numpy[0]['X']
#             Y_train = folds_data_numpy[0]['Y']

#             N_val = int(Y_train.shape[0]/10)

#             X_val = X_train[:N_val]
#             Y_val = Y_train[:N_val]
#             X_train = X_train[N_val:]
#             Y_train = Y_train[N_val:]
#             # print(X_train.shape,Y_train.shape)

#     else:
#         return AttributeError("incorrect evaluation_mode")

#     if scaler is not None:
#         X_train = scaler.transform(X_train)
#         X_test = scaler.transform(X_test)
#         X_val = scaler.transform(X_val)

#     X_train_up = X_train.copy()
#     Y_train_up = Y_train.copy()

#     # upsampling
#     if upsampling:
#         n_classes = Y_train.shape[1]
#         Ns = np.zeros(n_classes)
#         for j in range(n_classes):
#             Ns[j] = np.sum(Y_train[:, j] == 1)
#         Ns = np.floor(np.amax(Ns)/Ns)-1
#         for j in range(n_classes):
#             if Ns[j] > 1:
#                 X_j = X_train[Y_train[:, j] == 1]
#                 Y_j = Y_train[Y_train[:, j] == 1]
#                 X_train_up = np.concatenate(
#                     [X_train_up]+[X_j]*int(Ns[j]), axis=0)
#                 Y_train_up = np.concatenate(
#                     [Y_train_up]+[Y_j]*int(Ns[j]), axis=0)
#     return (X_train_up, Y_train_up, X_val, Y_val)


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


# def get_data_test(model, folds_data_test, fold_test, folds_files, scaler):
#     fold_list = list(folds_data_test.keys())

#     X_test = folds_data_test[fold_test]['X']
#     Y_test = folds_data_test[fold_test]['Y']

#     N_folds = len(folds_data_test)
#     if N_folds > 2:
#         fold_val = get_fold_val(fold_test, fold_list)

#         X_val = folds_data_test[fold_val]['X']
#         Y_val = folds_data_test[fold_val]['Y']
#         # print(len(X_val))
#         #X_val = scaler.transform(X_val)

#         folds_train = fold_list.copy()  # list(range(1,N_folds+1))
#         folds_train.remove(fold_test)
#         folds_train.remove(fold_val)
#         # print(folds_train)
#     else:
#         folds_train = [0]

#     #print('\nFold' + str(fold_test), 'Fold_val: ' + str(fold_val))
#     X_train = []
#     Y_train = []
#     Files_names_train = []
#     for fold_train in folds_train:
#         for file in range(len(folds_data_test[fold_train]['X'])):
#             X = folds_data_test[fold_train]['X'][file]
#             if len(X) <= 1:
#                 continue
#             ix = int(len(X)/2)
#             X = np.expand_dims(
#                 folds_data_test[fold_train]['X'][file][ix], axis=0)
#             # print(ix,X.shape)
#             X_train.append(X)
#             Y = np.expand_dims(
#               folds_data_test[fold_train]['Y'][file], axis=0
#             )
#             # print(Y.shape)
#             Y_train.append(Y)
#             if folds_files is not None:
#                 Files_names_train.append(folds_files[fold_train][file])
#     # print(Y_train[0],Y_train[1])
#     X_train = np.concatenate(X_train, axis=0)
#     Y_train = np.concatenate(Y_train, axis=0)
#     # print(X_train.shape,Y_train.shape)

#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)

#    # model_features = get_features(model)
#     #X_feat,_,_,_,_,_ = model_features.predict(X_train)
#     X_feat = None

#     return (X_feat, X_train, Y_train, Files_names_train)


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
