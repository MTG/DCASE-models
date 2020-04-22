import numpy as np

def accuracy(model, X_val, Y_val):
    """
    Calculates accuracy over files with different length

    Parameters
    ----------
    model : keras Model
        model to get the predictions
    X_val : list of ndarray
        Each element in list is a 3D array with the mel-spectrograms
        of one file. Shape of each element: (N_windows, N_hops, N_mel_bands)
        N_windows can be different in each file (element)
    Y_val : list ndarray
        Each element in the list is a 1D array with the annotations (one hot encoding).
        Shape of each elment (N_classes,)
    """

    n_files = len(X_val)

    predictions = np.zeros(n_files)
    annotations = np.zeros(n_files)

    for i in range(n_files):
        X = X_val[i]
        Y = Y_val[i]
        Y_predicted = model.predict(X)
        # if multiple outputs, select the first
        if type(Y_predicted) == list:
            Y_predicted = Y_predicted[0]
                
        Y_predicted = np.sum(Y_predicted,axis=0) #np.amax(Y_predicted,axis=0)
        Y_predicted = np.argmax(Y_predicted)
        Y = np.argmax(Y)

        annotations[i] = Y
        predictions[i] = Y_predicted 

    acc = np.mean(annotations==predictions)

    return acc, annotations, predictions