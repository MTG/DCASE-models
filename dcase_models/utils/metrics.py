# from scipy import interpolate
import numpy as np
from scipy.stats import mode
from dcase_models.utils.events import event_roll_to_event_list
from sed_eval.sound_event import SegmentBasedMetrics

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
        Each element in the list is a 1D array with
        the annotations (one hot encoding).
        Shape of each element (N_classes,)
    """
    n_files = len(X_val)

    predictions = []  # np.zeros(n_files)
    # annotations = np.zeros(n_files)

    for i in range(n_files):
        X = X_val[i]
        # Y = Y_val[i]
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


def sed(Y_val, Y_predicted, sequence_time_sec=0.5,
        metric_resolution_sec=1.0, label_list=[]):
    seg_metrics = SegmentBasedMetrics(
        label_list, time_resolution=metric_resolution_sec
    )

    n_files = len(Y_val)

    for i in range(n_files):
        y_true = Y_val[i]
        pred = Y_predicted[i]

        pred = (pred > 0.5).astype(int)
        event_list_val = event_roll_to_event_list(
            y_true, label_list, sequence_time_sec)
        event_list_pred = event_roll_to_event_list(
            pred, label_list, sequence_time_sec)

        seg_metrics.evaluate(event_list_val, event_list_pred)

    return seg_metrics


def ER(Y_val, Y_predicted, sequence_time_sec=0.5, metric_resolution_sec=1.0):
    n_files = len(Y_val)

    predictions = []
    annotations = []

    for i in range(n_files):
        y_true = Y_val[i]
        pred = Y_predicted[i]

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

    Sus = min(Nref, Nsys) - Ntp
    Del = max(0.0, Nref - Nsys)
    Ins = max(0.0, Nsys - Nref)

    ER = (Sus+Del+Ins)/float(Nref + eps)

    return ER


def F1(Y_val, Y_predicted, sequence_time_sec=0.5, metric_resolution_sec=1.0):
    n_files = len(Y_val)

    predictions = []
    annotations = []

    for i in range(n_files):
        y_true = Y_val[i]
        pred = Y_predicted[i]

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
    # Ntn = np.sum(predictions + annotations > 0)
    # Nfp = np.sum(predictions - annotations > 0)
    # Nfn = np.sum(annotations - predictions > 0)
    Nref = np.sum(annotations)
    Nsys = np.sum(predictions)

    P = Ntp / float(Nsys + eps)
    R = Ntp / float(Nref + eps)

    Fmeasure = 2*P*R/(P + R + eps)
    return Fmeasure
