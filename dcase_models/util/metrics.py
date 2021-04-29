# encoding: utf-8
"""Metric functions"""

# from scipy import interpolate
import numpy as np
from scipy.stats import mode
from dcase_models.util.events import event_roll_to_event_list
from dcase_models.util.events import tag_probabilities_to_tag_list
from sed_eval.sound_event import SegmentBasedMetrics
from sed_eval.scene import SceneClassificationMetrics
from sed_eval.audio_tag import AudioTaggingMetrics

eps = 1e-6


def predictions_temporal_integration(Y_predicted, type='sum'):
    """ Integrate temporal dimension.

    Parameters
    ----------
    Y_predicted : ndarray
        Signal to be integrated.
        e.g. shape (N_times, N_classes)
    type : str
        Type of integration ('sum', 'mean', 'autopool')

    Returns
    -------
    array
        Integrated signal.
        e.g. shape (N_classes,)

    """
    if type == 'sum':
        Y_predicted = np.sum(Y_predicted, axis=0)
    if type == 'max':
        Y_predicted = np.max(Y_predicted, axis=0)
    if type == 'mode':
        Y_predicted, _ = mode(Y_predicted, axis=0)
        Y_predicted = np.squeeze(Y_predicted, axis=0)
    return Y_predicted


def evaluate_metrics(model, data, metrics, **kwargs):
    """ Calculate metrics over files with different length

    Parameters
    ----------
    model : keras Model
        model to get the predictions
    data : tuple or KerasDataGenerator
        Validation data for model evaluation
        (X_val, Y_val) or KerasDataGenerator

        X_val : list of ndarray
            Each element in list is a 3D array with the mel-spectrograms
            of one file. Shape of each element:
            (N_windows, N_hops, N_mel_bands)
            N_windows can be different in each file (element)
        Y_val : list ndarray
            Each element in the list is a 1D array with
            the annotations (one hot encoding).
            Shape of each element (N_classes,)

    metrics : list
        List of metrics to apply.
        Each element can be a metric name or a function.

    Returns
    -------
    dict
        Dict with the results information.

        {'annotations' : [Y0, Y1, ...],
         'predictions' : [Yp0, Yp1, ...],
         metrics[0]: 0.1,
         metrics[1]: 0.54}

    """
    predictions = []
    annotations = []
    results = {}

    if type(data) in [list, tuple]:
        X_val = data[0]
        Y_val = data[1]
        n_files = len(X_val)
        for i in range(n_files):
            X = X_val[i]
            Y_predicted = model.predict(X)
            # if multiple outputs, select the first
            if type(Y_predicted) == list:
                Y_predicted = Y_predicted[0]
            predictions.append(Y_predicted)

        annotations = Y_val

    else:
        # data type is DataGenerator
        for batch_index in range(0, len(data)):
            X_val, Y_val = data.get_data_batch(batch_index)
            n_files = len(X_val)
            for i in range(n_files):
                X = X_val[i]
                Y_predicted = model.predict(X)
                if type(Y_predicted) == list:
                    Y_predicted = Y_predicted[0]
                predictions.append(Y_predicted)

            annotations.extend(Y_val)

    results['annotations'] = annotations
    results['predictions'] = predictions

    for metric in metrics:
        if callable(metric):
            metric_function = metric
        else:
            metric_function = globals()[metric]

        results[metric] = metric_function(annotations, predictions, **kwargs)
    return results

def _check_lists_for_evaluation(Y_val, Y_predicted):
    """ Perform the following checks
        1) Y_val and Y_predicted are both of type list
        2) Y_val and Y_predicted are of the same length
        3) Each element in Y_val and Y_predicted is a 2D array

    Parameters
    ----------
    Y_val : list of ndarray
        2D array with the ground-truth event roll
        shape: (N_times, N_classes)
    Y_predicted : list of ndarray
        2D array with the predicted event roll
        shape: (N_times, N_classes)
    sequence_time_sec : float
        Resolution of Y_val and Y_predicted.
    metric_resolution_sec : float
        Resolution of the metrics.
    label_list:
        Label list.

    Returns
    -------
    bool
        True if checks passed.

    """
    
    if type(Y_val) is not list:
        raise AttributeError(
            'Y_val type is invalid. It should be a list of 2D array and received {}'.format(
                type(Y_val)
            )
        )

    if type(Y_predicted) is not list:
        raise AttributeError(
            'Y_predicted type is invalid. It should be a list of 2D array and received {}'.format(
                type(Y_predicted)
            )
        )

    if len(Y_val) != len(Y_predicted):
        raise AttributeError('Y_val and Y_predicted should have the same length (received {:d} and {:d})'.format(
            len(Y_val), len(Y_predicted) 
            )
        )

    for j in range(len(Y_val)):
        if type(Y_val[j]) is not np.ndarray:
            raise AttributeError('Each element of Y_val should be a 2D numpy array and received {}'.format(
                type(Y_val[j])
            )
        )
        if len(Y_val[j].shape) != 2:
            raise AttributeError('Each element of Y_val should be a 2D array and received an array of shape {}'.format(
                str(Y_val[j].shape)
            )
        )
        if type(Y_predicted[j]) is not np.ndarray:
            raise AttributeError('Each element of Y_predicted should be a 2D numpy array and received {}'.format(
                type(Y_predicted[j])
            )
        )
        if len(Y_predicted[j].shape) != 2:
            raise AttributeError('Each element of Y_predicted should be a 2D array and received an array of shape {}'.format(
                str(Y_predicted[j].shape)
            )
        )

def sed(Y_val, Y_predicted, sequence_time_sec=0.5,
        metric_resolution_sec=1.0, label_list=[]):
    """ Calculate metrics for Sound Event Detection

    Parameters
    ----------
    Y_val : list of ndarray
        2D array with the ground-truth event roll
        shape: (N_times, N_classes)
    Y_predicted : list of ndarray
        2D array with the predicted event roll
        shape: (N_times, N_classes)
    sequence_time_sec : float
        Resolution of Y_val and Y_predicted.
    metric_resolution_sec : float
        Resolution of the metrics.
    label_list:
        Label list.

    Returns
    -------
    sef_eval.sound_events.SegmentBasedMetrics
        Object with the SED results

    """

    _check_lists_for_evaluation(Y_val, Y_predicted)

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


def classification(Y_val, Y_predicted, label_list=[]):
    """ Calculate metrics for Audio Classification

    Parameters
    ----------
    Y_val : listy of ndarray
        2D array with the ground-truth event roll
        shape: (N_times, N_classes)
    Y_predicted : list of ndarray
        2D array with the predicted event roll
        shape: (N_times, N_classes)
    label_list:
        Label list.

    Returns
    -------
    sef_eval.scenes.SceneClassificationMetrics
        Object with the classification results

    """
    _check_lists_for_evaluation(Y_val, Y_predicted)

    acc_metrics = SceneClassificationMetrics(label_list)

    n_files = len(Y_val)
    for i in range(n_files):
        y_true = Y_val[i]
        pred = Y_predicted[i]
        pred = np.sum(pred, axis=0)
        pred = np.argmax(pred)
        label_predicted = label_list[pred]
        label_gt = label_list[np.argmax(y_true[0])]
        acc_metrics.evaluate(
            [{'scene_label': label_gt, 'file': ''}],
            [{'scene_label': label_predicted, 'file': ''}])

    return acc_metrics


def tagging(Y_val, Y_predicted, label_list=[]):
    """ Calculate metrics for Audio Tagging

    Parameters
    ----------
    Y_val : list of ndarray
        2D array with the ground-truth event roll
        shape: (N_times, N_classes)
    Y_predicted : list of ndarray
        2D array with the predicted event roll
        shape: (N_times, N_classes)
    label_list:
        Label list.

    Returns
    -------
    sef_eval.scenes.AudioTaggingMetrics
        Object with the tagging results

    """
    _check_lists_for_evaluation(Y_val, Y_predicted)

    tagging_metrics = AudioTaggingMetrics(label_list)

    n_files = len(Y_val)

    for i in range(n_files):
        y_true = Y_val[i]
        pred = Y_predicted[i]
        pred = np.mean(pred, axis=0)

        tag_list_val = tag_probabilities_to_tag_list(
            y_true[0], label_list, threshold=0.5)
        tag_list_pred = tag_probabilities_to_tag_list(
            pred, label_list, threshold=0.5)

        tagging_metrics.evaluate(
            [{'tags': tag_list_val, 'file': ''}],
            [{'tags': tag_list_pred, 'file': ''}])

    return tagging_metrics
