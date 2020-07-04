# encoding: utf-8
"""Events functions"""

import numpy as np

# From Salamon's code
# https://github.com/justinsalamon/scaper_waspaa2017/blob/master/urban_sed/util.py


def contiguous_regions(act):
    act = np.asarray(act)
    onsets = np.where(np.diff(act) == 1)[0] + 1
    offsets = np.where(np.diff(act) == -1)[0] + 1

    # SPECIAL CASES
    # If there are no onsets and no offsets (all of act is the same value)
    if len(onsets) == 0 and len(offsets) == 0:
        if act[0] == 0:
            return np.asarray([])
        else:
            return np.asarray([[0, len(act)]])

    # If there are no onsets
    if len(onsets) == 0 and len(offsets) != 0:
        onsets = np.insert(onsets, 0, 0)

    # If there are no offsets
    if len(onsets) != 0 and len(offsets) == 0:
        offsets = np.insert(offsets, len(offsets), len(act))

    # If there's an onset before an offset, first onset is frame 0
    if onsets[0] > offsets[0]:
        onsets = np.insert(onsets, 0, 0)

    # If there's an onset after the last offset, then we need to add an offset
    # Offset is last index of activation (so that gives inverse of sed_eval)
    if onsets[-1] > offsets[-1]:
        offsets = np.insert(offsets, len(offsets), len(act))

    assert len(onsets) == len(offsets)
    assert (onsets <= offsets).all()
    return np.asarray([onsets, offsets]).T

# From Salamon's code
# https://github.com/justinsalamon/scaper_waspaa2017/blob/master/urban_sed/util.py


def event_roll_to_event_list(event_roll, event_label_list, time_resolution):
    """ Convert a event roll matrix to a event list.

    Parameters
    ----------
    event_roll : ndarray
        Shape (N_times, N_classes)
    event_label_list : list of str
        Label list
    time_resolution : float
        Time resolution of the event_roll.

    Returns
    -------
    list
        List of dicts with events information.
        e.g. 

            [{'event_onset': 0.1,
              'event_offset': 1.5,
              'event_label' : 'dog'}, ...]

    """
    event_list = []
    for event_id, event_label in enumerate(event_label_list):
        event_activity = event_roll[:, event_id]
        event_segments = contiguous_regions(event_activity) * time_resolution
        for event in event_segments:
            event_list.append(
                    {'event_onset': event[0],
                     'event_offset': event[1],
                     'event_label': event_label})

    return event_list


def tag_probabilities_to_tag_list(tag_probabilities, label_list,
                                  threshold=0.5):
    """ Convert a tag probabilites matrix to a tag list.

    Parameters
    ----------
    tag_probabilities : ndarray
        Shape (N_times, N_classes)
    label_list : list of str
        Label list
    threshold : float
        Threshold to decide if a tag is present.

    Returns
    -------
    list
        List of tags.
        e.g. ['dog', 'cat', ...]

    """
    tag_binary = (tag_probabilities > threshold).astype(int)
    tag_indexes = np.argwhere(tag_binary == 1)
    tag_list = [label_list[index[0]] for index in tag_indexes]

    return tag_list
