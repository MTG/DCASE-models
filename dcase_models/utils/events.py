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