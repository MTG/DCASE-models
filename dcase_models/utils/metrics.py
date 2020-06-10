from scipy import interpolate
import numpy as np
from scipy.stats import mode

eps = 1e-6


def predictions_temporal_integration(Y_predicted, type='sum'):
    if type == 'sum':
        Y_predicted = np.sum(Y_predicted, axis=0)
    if type == 'max':
        Y_predicted = np.max(Y_predicted, axis=0)
    if type == 'mode':
        Y_predicted, _ = mode(Y_predicted, axis=0)
        Y_predicted = np.squeeze(Y_predicted, axis=0)
    return Y_predicted


def evaluate_metrics(model, X_val, Y_val, metrics, **kwargs):
    """
    Calculates metrics over files with different length

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

    predictions = []  # np.zeros(n_files)
    #annotations = np.zeros(n_files)

    for i in range(n_files):
        X = X_val[i]
        Y = Y_val[i]
        Y_predicted = model.predict(X)
        # if multiple outputs, select the first
        if type(Y_predicted) == list:
            Y_predicted = Y_predicted[0]
        predictions.append(Y_predicted)

    results = {}
    results['annotations'] = Y_val
    results['predictions'] = predictions
    for metric in metrics:
        if callable(metric):
            metric_function = metric
        else:
            metric_function = globals()[metric]

        results[metric] = metric_function(Y_val, predictions, **kwargs)
    return results


def accuracy(Y_val, Y_predicted):
    n_files = len(Y_val)

    predictions = np.zeros(n_files)
    annotations = np.zeros(n_files)

    for i in range(n_files):
        Y = Y_val[i]
        pred = predictions_temporal_integration(Y_predicted[i], 'sum')
        pred = np.argmax(pred)
        Y = np.argmax(Y)
        annotations[i] = Y
        predictions[i] = pred

    acc = np.mean(annotations == predictions)

    return acc


def ER(Y_val, Y_predicted, sequence_time_sec=0.5, metric_resolution_sec=1.0):
    n_files = len(Y_val)

    predictions = []
    annotations = []

    for i in range(n_files):
        y_true = Y_val[i]
        pred = Y_predicted[i]
        # change resolution
        #print(y_true.shape, pred.shape)
  #      time_grid_Y = np.linspace(0, y_true.shape[0]*metric_resolution_sec, y_true.shape[0])
  #      time_grid_pred = np.linspace(0, pred.shape[0]*sequence_time_sec, pred.shape[0])
  #      print(time_grid_pred.shape, pred.shape)
  #      print(time_grid_Y)
  #      print(time_grid_pred)
 #       f = interpolate.interp1d(time_grid_pred, pred, axis=0)
  #      y_pred = f(time_grid_Y)
        if pred.shape[0] == y_true.shape[0]:
            y_pred = pred
        else:
            y_pred = np.zeros_like(y_true)
            ratio = int(np.round(metric_resolution_sec / sequence_time_sec))
            for j in range(len(y_true)):
                y_pred[j] = np.mean(pred[j*ratio:(j+1)*ratio], axis=0)

        annotations.append(y_true)
        predictions.append(y_pred)

    annotations = np.concatenate(annotations, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    assert annotations.shape[0] == predictions.shape[0]
    assert annotations.shape[1] == predictions.shape[1]

    predictions = (predictions > 0.5).astype(int)
    Ntp = np.sum(predictions + annotations > 1)
    Nref = np.sum(annotations)
    Nsys = np.sum(predictions)

    S = min(Nref, Nsys) - Ntp
    D = max(0.0, Nref - Nsys)
    I = max(0.0, Nsys - Nref)

    ER = (S+D+I)/float(Nref + eps)

    return ER


def F1(Y_val, Y_predicted, sequence_time_sec=0.5, metric_resolution_sec=1.0):
    n_files = len(Y_val)

  #  print(Y_val[0].shape,Y_predicted[0].shape)

    predictions = []
    annotations = []

    for i in range(n_files):
        y_true = Y_val[i]
        pred = Y_predicted[i]
        # change resolution
   #     print(y_true.shape, pred.shape)
  #      time_grid_Y = np.linspace(0, y_true.shape[0]*metric_resolution_sec, y_true.shape[0])
  #      time_grid_pred = np.linspace(0, pred.shape[0]*sequence_time_sec, pred.shape[0])
  #      print(time_grid_pred.shape, pred.shape)
  #      print(time_grid_Y)
  #      print(time_grid_pred)
 #       f = interpolate.interp1d(time_grid_pred, pred, axis=0)
  #      y_pred = f(time_grid_Y)
        if pred.shape[0] == y_true.shape[0]:
            y_pred = pred
        else:
            y_pred = np.zeros_like(y_true)
            ratio = int(np.round(metric_resolution_sec / sequence_time_sec))
            for j in range(len(y_true)):
                y_pred[j] = np.mean(pred[j*ratio:(j+1)*ratio], axis=0)

        annotations.append(y_true)
        predictions.append(y_pred)

    annotations = np.concatenate(annotations, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    assert annotations.shape[0] == predictions.shape[0]
    assert annotations.shape[1] == predictions.shape[1]

    predictions = (predictions > 0.5).astype(int)
    Ntp = np.sum(predictions + annotations > 1)
    Ntn = np.sum(predictions + annotations > 0)
    Nfp = np.sum(predictions - annotations > 0)
    Nfn = np.sum(annotations - predictions > 0)
    Nref = np.sum(annotations)
    Nsys = np.sum(predictions)

    P = Ntp / float(Nsys + eps)
    R = Ntp / float(Nref + eps)

    Fmeasure = 2*P*R/(P + R + eps)
    return Fmeasure
