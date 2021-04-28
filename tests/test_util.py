from dcase_models.util.data import get_fold_val, evaluation_setup
from dcase_models.util.events import (
    contiguous_regions,
    event_roll_to_event_list,
    tag_probabilities_to_tag_list,
)
from dcase_models.util.files import save_pickle, load_pickle
from dcase_models.util.files import save_json, load_json, mkdir_if_not_exists
from dcase_models.util.files import (
    duplicate_folder_structure,
    list_wav_files,
    list_all_files,
)
from dcase_models.util.files import load_training_log, download_files_and_unzip
from dcase_models.util.files import move_all_files_to_parent, move_all_files_to
from dcase_models.util.files import example_audio_file
from dcase_models.util.misc import get_class_by_name, get_default_args_of_function
from dcase_models.util.metrics import predictions_temporal_integration
from dcase_models.util.metrics import sed, classification, tagging, evaluate_metrics
from dcase_models.util.callbacks import (
    ClassificationCallback,
    SEDCallback,
    TaggingCallback,
)

import os
import numpy as np
import pytest
import glob
import shutil
import csv
from sed_eval.util.event_roll import event_list_to_event_roll
from sed_eval.sound_event import SegmentBasedMetrics
from sed_eval.scene import SceneClassificationMetrics
from sed_eval.audio_tag import AudioTaggingMetrics


def _clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


def test_get_fold_val():
    fold_list = ["fold1", "fold2", "fold3"]
    assert get_fold_val("fold1", fold_list) == "fold2"
    assert get_fold_val("fold2", fold_list) == "fold3"
    assert get_fold_val("fold3", fold_list) == "fold1"


def test_evaluation_setup():
    fold_list = ["fold1", "fold2", "fold3", "fold4"]
    fold = "fold1"
    evaluation_mode = "cross-validation"

    folds_train, folds_val, folds_test = evaluation_setup(
        fold, fold_list, evaluation_mode, use_validate_set=True
    )

    assert folds_test == ["fold1"]
    assert folds_val == ["fold2"]
    assert folds_train == ["fold3", "fold4"]

    folds_train, folds_val, folds_test = evaluation_setup(
        fold, fold_list, evaluation_mode, use_validate_set=False
    )

    assert folds_test == ["fold1"]
    assert folds_val == folds_train
    assert folds_train == ["fold2", "fold3", "fold4"]

    evaluation_mode = "train-validate-test"

    folds_train, folds_val, folds_test = evaluation_setup(
        fold, fold_list, evaluation_mode, use_validate_set=True
    )

    assert folds_test == ["test"]
    assert folds_val == ["validate"]
    assert folds_train == ["train"]

    evaluation_mode = "train-test"

    folds_train, folds_val, folds_test = evaluation_setup(
        fold, fold_list, evaluation_mode, use_validate_set=True
    )

    assert folds_test == ["test"]
    assert folds_val == ["train"]
    assert folds_train == ["train"]

    evaluation_mode = "cross-validation-with-test"

    folds_train, folds_val, folds_test = evaluation_setup(
        fold, fold_list, evaluation_mode, use_validate_set=True
    )

    assert folds_test == ["test"]
    assert folds_val == ["fold2"]
    assert folds_train == ["fold1", "fold3", "fold4"]

    evaluation_mode = "blablabla"
    with pytest.raises(AttributeError):
        folds_train, folds_val, folds_test = evaluation_setup(
            fold, fold_list, evaluation_mode, use_validate_set=True
        )


def test_contiguous_regions():
    onsets = [2, 10, 20, 33]
    offsets = [5, 15, 22, 39]
    activation = np.zeros(50)
    for on, off in zip(onsets, offsets):
        activation[on:off] = 1

    cr = contiguous_regions(activation)
    assert cr.shape == (4, 2)
    assert np.allclose(onsets, cr[:, 0])
    assert np.allclose(offsets, cr[:, 1])

    activation = np.ones(50)
    cr = contiguous_regions(activation)
    assert np.allclose(cr, np.asarray([[0, 50]]))

    activation = np.zeros(50)
    cr = contiguous_regions(activation)
    assert len(cr) == 0

    onsets = [0]
    offsets = [5]
    activation = np.zeros(50)
    for on, off in zip(onsets, offsets):
        activation[on:off] = 1

    cr = contiguous_regions(activation)
    assert cr.shape == (1, 2)
    assert np.allclose(onsets, cr[:, 0])
    assert np.allclose(offsets, cr[:, 1])

    onsets = [5]
    offsets = [50]
    activation = np.zeros(50)
    for on, off in zip(onsets, offsets):
        activation[on:off] = 1

    cr = contiguous_regions(activation)
    assert cr.shape == (1, 2)
    assert np.allclose(onsets, cr[:, 0])
    assert np.allclose(offsets, cr[:, 1])


def test_event_roll_to_event_list():
    onsets = [2, 10, 20, 33]
    offsets = [5, 15, 22, 39]
    event_label_list = ["class1"]
    time_resolution = 2.0
    event_roll = np.zeros((50, 1))
    for on, off in zip(onsets, offsets):
        event_roll[on:off, 0] = 1

    event_list = event_roll_to_event_list(event_roll, event_label_list, time_resolution)
    assert len(event_list) == 4
    for j, event in enumerate(event_list):
        event["event_onset"] == onsets[j] * time_resolution
        event["event_offset"] == offsets[j] * time_resolution
        event["event_label"] == "class1"


def test_tag_probabilities_to_tag_list():
    tags = ["tag1", "tag2", "tag3"]
    n_tags = len(tags)
    tag_probabilities = np.zeros(n_tags)
    tag_probabilities[0] = 0.7
    tag_probabilities[2] = 0.75
    tag_list = tag_probabilities_to_tag_list(tag_probabilities, tags, threshold=0.5)

    assert tag_list == ["tag1", "tag3"]


def test_save_pickle():
    filename = "./test.pickle"
    obj = [np.zeros((2, 1))]
    _clean(filename)
    save_pickle(obj, filename)
    assert os.path.exists(filename)
    _clean(filename)


def test_load_pickle():
    filename = "./test.pickle"
    obj = [np.zeros((2, 1))]
    _clean(filename)
    save_pickle(obj, filename)
    obj_loaded = load_pickle(filename)
    assert type(obj_loaded) is list
    assert len(obj_loaded) == 1
    assert np.allclose(obj, obj_loaded)
    _clean(filename)


def test_save_json():
    filename = "./test.json"
    obj = {"test1": "yes", "test2": "no"}
    _clean(filename)
    save_json(filename, obj)
    assert os.path.exists(filename)
    _clean(filename)


def test_load_json():
    filename = "./test.json"
    obj = {"test1": "yes", "test2": "no"}
    _clean(filename)
    save_json(filename, obj)
    obj_loaded = load_json(filename)
    assert type(obj_loaded) is dict
    assert obj == obj_loaded
    _clean(filename)


def test_mkdir_if_not_exists():
    folder = "./new_folder"
    _clean(folder)
    mkdir_if_not_exists(folder)
    assert os.path.isdir(folder)
    _clean(folder)

    folder = "./new_parent/new_folder"
    mkdir_if_not_exists(folder, parents=True)
    assert os.path.isdir("./new_parent")
    assert os.path.isdir(folder)
    _clean(folder)
    _clean("./new_parent")


def test_duplicate_folder_structure():
    folder1 = "./new_parent/new_folder1"
    folder2 = "./new_parent/new_folder2"
    mkdir_if_not_exists(folder1, parents=True)
    mkdir_if_not_exists(folder2)

    folder_destination = "./test_dest"
    _clean(folder_destination)
    duplicate_folder_structure("./new_parent", folder_destination)
    assert os.path.isdir(folder_destination)
    assert os.path.isdir(os.path.join(folder_destination, "new_folder1"))
    assert os.path.isdir(os.path.join(folder_destination, "new_folder2"))
    _clean(folder_destination)
    _clean("./new_parent")


def test_list_wav_files():
    audio_files = ["147764-4-7-0.wav", "176787-5-0-0.wav", "40722-8-0-7.wav"]
    audio_path = "./tests/data/audio"
    wav_files = list_wav_files(audio_path)
    assert type(wav_files) is list
    assert len(wav_files) == 3
    for wf in audio_files:
        wp = os.path.join(audio_path, wf)
        assert wp in wav_files

    audio_path = "./tests/data_aiff/audio"
    wav_files = list_wav_files(audio_path)
    assert len(wav_files) == 0


def test_list_all_files():
    aiff_files = ["147764-4-7-0.aiff", "176787-5-0-0.aiff", "40722-8-0-7.aiff"]
    audio_path = "./tests/data_aiff/audio"
    all_files = list_all_files(audio_path)
    assert type(all_files) is list
    assert len(all_files) == 3
    for ff in aiff_files:
        wp = os.path.join(audio_path, ff)
        assert wp in all_files


def test_load_training_log():
    path = "./tests/resources"
    log = load_training_log(path)
    assert type(log) is dict
    assert len(log) == 3

    epoch = ["0", "1", "2"]
    Acc = ["0.2927927927927928", "0.3141891891891892", "0.3536036036036036"]
    loss = ["1.9091888414991058", "1.5893025775367684", "1.452262936726055"]

    assert Acc == log["Acc"]
    assert epoch == log["epoch"]
    assert loss == log["loss"]

    path = "./tests/resources2"
    log = load_training_log(path)
    assert log is None


def test_download_files_and_unzip():
    dataset_path = "./tests/data"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    url = "file:////" + os.path.join(dir_path, "resources")
    files = ["remote.zip"]
    unzip_file = os.path.join(dataset_path, "remote.wav")
    _clean(unzip_file)
    download_files_and_unzip(dataset_path, url, files)
    assert os.path.exists(unzip_file)
    _clean(unzip_file)


def test_move_all_files_to_parent():
    parent = "./new_parent"
    mkdir_if_not_exists("./new_parent/new_child", parents=True)
    mkdir_if_not_exists("./new_parent/new_child2")
    move_all_files_to_parent("./", "new_parent")
    assert os.path.exists("./new_child")
    assert os.path.exists("./new_child2")
    _clean("./new_child")
    _clean("./new_child2")
    _clean("./new_parent")


def test_move_all_files_to():
    parent = "./new_parent"
    mkdir_if_not_exists("./new_parent/new_child", parents=True)
    mkdir_if_not_exists("./new_parent/new_child2")
    dest = "./dest"
    move_all_files_to("./new_parent", dest)
    assert os.path.exists("./dest/new_child")
    assert os.path.exists("./dest/new_child2")
    assert not os.path.exists("./new_parent/new_child")
    assert not os.path.exists("./new_parent/new_child2")
    _clean("./dest")
    _clean("./new_parent")


def test_example_audio_file():
    audio_file = example_audio_file()
    assert type(audio_file) is str
    assert os.path.exists(audio_file)


def test_get_class_by_name():
    class Class1:
        def __init__():
            pass

    class Class2:
        def __init__():
            pass

    class Default:
        def __init__():
            pass

    classes_dict = {"Class1": Class1, "Class2": Class2}

    class_r = get_class_by_name(classes_dict, "Class1", Default)
    assert class_r == Class1

    class_r = get_class_by_name(classes_dict, "Class2", Default)
    assert class_r == Class2

    class_r = get_class_by_name(classes_dict, "Class3", Default)
    assert class_r == Default

    class_r = get_class_by_name(classes_dict, "Class1_new", Default)
    assert class_r == Class1


def test_get_default_args_of_function():
    def fun(arg1=1, arg2=2, arg3=None):
        pass

    args = get_default_args_of_function(fun)
    assert type(args) is dict
    assert len(args) == 3
    assert args["arg1"] == 1
    assert args["arg2"] == 2
    assert args["arg3"] == None


def test_predictions_temporal_integration():
    pred = np.zeros((10, 3))
    pred[:, 0] = 1
    pred[3, 1] = 1
    pred[5, 2] = 1
    pred[6, 2] = 2

    pred_int = predictions_temporal_integration(pred, type="sum")
    assert np.allclose(pred_int, [10, 1, 3])

    pred_int = predictions_temporal_integration(pred, type="max")
    assert np.allclose(pred_int, [1, 1, 2])

    pred_int = predictions_temporal_integration(pred, type="mode")
    assert np.allclose(pred_int, [1, 0, 0])


def test_sed():
    with open("./tests/resources/sed_example.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        event_list = []
        for row in csv_reader:
            event_list.append(
                {
                    "event_onset": float(row[1]),
                    "event_offset": float(row[2]),
                    "event_label": row[0],
                }
            )
    time_resolution = 1.0
    label_list = ["class1", "class2", "class3"]
    event_roll = event_list_to_event_roll(event_list, label_list, time_resolution)
    event_roll_pred = event_roll * np.random.uniform(0.5, 1.0, size=event_roll.shape)
    metrics = sed([event_roll], [event_roll_pred], label_list=label_list)
    assert metrics.results()["overall"]["f_measure"]["f_measure"] == 1.0

    seg_metrics = SegmentBasedMetrics(label_list, time_resolution=time_resolution)
    event_pred = event_roll_to_event_list(event_roll, label_list, time_resolution)
    seg_metrics.evaluate(event_list, event_pred)

    assert metrics.results()["overall"]["f_measure"]["f_measure"] == 1.0


def test_classification():
    label_list = ["class1", "class2", "class3"]
    Y_pred = np.zeros((10, 3))

    Y_pred[:, 0] = 1
    Y_pred[1:2, 1] = 1
    Y_pred[4:7, 1] = 1

    metrics = classification([Y_pred], [Y_pred], label_list)
    assert metrics.results()["overall"]["accuracy"] == 1.0

    with pytest.raises(AttributeError):
        metrics = classification(Y_pred, Y_pred, label_list)

    with pytest.raises(AttributeError):
        metrics = classification([Y_pred], Y_pred, label_list)

    with pytest.raises(AttributeError):
        metrics = classification(Y_pred, [Y_pred], label_list)

    with pytest.raises(AttributeError):
        metrics = classification([Y_pred], [Y_pred, Y_pred], label_list)

    with pytest.raises(AttributeError):
        metrics = classification([Y_pred, Y_pred], [Y_pred, None], label_list)

    with pytest.raises(AttributeError):
        metrics = classification([Y_pred, None], [Y_pred, Y_pred], label_list)

    fake_pred = np.zeros((2, 2, 2))

    with pytest.raises(AttributeError):
        metrics = classification([Y_pred, fake_pred], [Y_pred, Y_pred], label_list)

    with pytest.raises(AttributeError):
        metrics = classification([Y_pred, Y_pred], [Y_pred, fake_pred], label_list)


def test_tagging():
    label_list = ["class1", "class2", "class3"]
    Y_pred = np.zeros((10, 3))

    Y_pred[:, 0] = 1
    Y_pred[1:2, 1] = 1
    Y_pred[4:7, 1] = 1

    metrics = tagging([Y_pred], [Y_pred], label_list)
    assert metrics.results()["overall"]["f_measure"]["f_measure"] == 1.0


def test_evaluate_metrics():
    class ToyModel:
        def __init__(self, Y_val):
            self.Y_val = Y_val

        def predict(self, X_val):
            return self.Y_val

    Y_val = np.zeros((10, 3))
    Y_val[:, 0] = 1
    Y_val[1:2, 1] = 1
    Y_val[4:7, 1] = 1
    X_val = np.zeros((10, 10))
    toy_model = ToyModel(Y_val)
    label_list = ["class1", "class2", "class3"]
    metrics = ["classification", "sed", "tagging"]
    results = evaluate_metrics(
        toy_model, ([X_val], [Y_val]), metrics, label_list=label_list
    )

    assert type(results) is dict
    assert len(results) == 5
    assert len(results["annotations"]) == 1
    assert np.allclose(results["annotations"], Y_val)
    assert np.allclose(results["predictions"], Y_val)
    assert results["tagging"].results()["overall"]["f_measure"]["f_measure"] == 1.0
    assert results["sed"].results()["overall"]["f_measure"]["f_measure"] == 1.0
    assert results["classification"].results()["overall"]["accuracy"] == 1.0

    # add custom metric
    def custom_metric(Y_val, Y_predicted, **kwargs):
        return np.sum(Y_val)

    metrics = [custom_metric]
    results = evaluate_metrics(
        toy_model, ([X_val], [Y_val]), metrics, label_list=label_list
    )

    assert type(results) is dict
    assert len(results) == 3
    assert len(results["annotations"]) == 1
    assert np.allclose(results["annotations"], Y_val)
    assert np.allclose(results["predictions"], Y_val)
    assert results[custom_metric] == np.sum(Y_val)

    # multi-output
    class ToyModel:
        def __init__(self, Y_val):
            self.Y_val = Y_val

        def predict(self, X_val):
            return [self.Y_val, 0, 1]

    toy_model = ToyModel(Y_val)
    results = evaluate_metrics(
        toy_model, ([X_val], [Y_val]), metrics, label_list=label_list
    )
    assert type(results) is dict
    assert len(results) == 3
    assert len(results["annotations"]) == 1
    assert np.allclose(results["annotations"][0], Y_val)
    assert np.allclose(results["predictions"][0], Y_val)
    assert results[custom_metric] == np.sum(Y_val)

    class ToyDataGenerator:
        def __init__(self, X_val, Y_val):
            self.X_val = X_val
            self.Y_val = Y_val

        def __len__(self):
            return 3

        def get_data_batch(self, index):
            return [X_val], [Y_val]

    toy_data_gen = ToyDataGenerator(X_val, Y_val)

    results = evaluate_metrics(toy_model, toy_data_gen, metrics, label_list=label_list)
    assert type(results) is dict
    assert len(results) == 3
    assert len(results["annotations"]) == 3
    assert len(results["predictions"]) == 3
    assert np.allclose(results["annotations"][0], Y_val)
    assert np.allclose(results["predictions"][0], Y_val)


callbacks = [ClassificationCallback, SEDCallback, TaggingCallback]


@pytest.mark.parametrize("callback_class", callbacks)
def test_callbacks(callback_class):
    class ToyModel:
        def __init__(self, Y_val):
            self.Y_val = Y_val
            self.weights_saved = False
            self.stop_training = False

        def predict(self, X_val):
            return self.Y_val

        def save_weights(self, file_weights):
            self.weights_saved = True

    Y_val = np.zeros((10, 3))
    Y_val[:, 0] = 1
    Y_val[1:2, 1] = 1
    Y_val[4:7, 1] = 1
    X_val = np.zeros((10, 10))
    toy_model = ToyModel(Y_val)

    label_list = ["class1", "class2", "class3"]

    callback = callback_class(
        ([X_val], [Y_val]),
        file_weights="",
        early_stopping=3,
        considered_improvement=0.01,
        label_list=label_list,
    )

    callback.model = toy_model
    callback.on_epoch_end(0)
    assert toy_model.weights_saved
    toy_model.weights_saved = False
    callback.on_epoch_end(1)
    assert not toy_model.weights_saved

    callback.on_epoch_end(2)
    assert toy_model.stop_training
