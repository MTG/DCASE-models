# encoding: utf-8
"""
Utilities
=========

Metric functions
----------------
.. autosummary::
    :toctree: generated/

    predictions_temporal_integration
    evaluate_metrics
    sed
    classification
    tagging
    accuracy
    ER
    F1

Data functions
--------------
.. autosummary::
    :toctree: generated/

    get_fold_val
    evaluation_setup
    

Events functions
----------------
.. autosummary::
    :toctree: generated/

    contiguous_regions
    evaluation_setup
    event_roll_to_event_list
    tag_probabilities_to_tag_list

Files functions
---------------
.. autosummary::
    :toctree: generated/

    save_json
    load_json
    save_pickle
    load_pickle
    list_all_files
    list_wav_files
    load_training_log
    mkdir_if_not_exists
    download_files_and_unzip
    move_all_files_to
    move_all_files_to_parent
    duplicate_folder_structure
    example_audio_file
    
Callback functions
------------------
.. autosummary::
    :toctree: generated/

    ClassificationCallback
    SEDCallback
    TaggingCallback
    F1ERCallback

GUI functions
-------------
.. autosummary::
    :toctree: generated/

    encode_audio

UI functions
------------
.. autosummary::
    :toctree: generated/

    progressbar

Miscellaneous functions
-----------------------
.. autosummary::
    :toctree: generated/

    get_class_by_name

"""

from .callbacks import *  # pylint: disable=wildcard-import
from .data import *  # pylint: disable=wildcard-import
from .events import *  # pylint: disable=wildcard-import
from .files import *  # pylint: disable=wildcard-import
from .gui import *  # pylint: disable=wildcard-import
from .metrics import *  # pylint: disable=wildcard-import
from .misc import *  # pylint: disable=wildcard-import
from .ui import *  # pylint: disable=wildcard-import


