backends = []

try:
    import tensorflow as tf

    tensorflow_version = '2' if tf.__version__.split(".")[0] == "2" else '1'

    if tensorflow_version == '2':
        backends.append('tensorflow2')
    else:
        backends.append('tensorflow1')
except:
    tensorflow_version = None

try:
    import torch
    backends.append('torch')
except:
    torch = None

try:
    import sklearn
    backends.append('sklearn')
except:
    sklearn = None
