# encoding: utf-8
"""GUI functions"""

from io import BytesIO
import wave
import struct
import numpy as np


def encode_audio(data, sr):
    """ Encode an audio signal for web applications.

    Parameters
    ----------
    data : array
        Audio signal.
    sr : int
        Sampling rate

    Returns
    -------
    str
        Encoded audio signal.

    """
    import base64
    if len(data.shape) == 1:
        n_channels = 1
        data_audio = data
    else:
        n_channels = data.shape[1]
        data_audio = data.ravel()

    data_audio = data_audio/np.amax(data_audio)
    data_audio = np.int16(data_audio * 32767).tolist()

    fp = BytesIO()
    waveobj = wave.open(fp, mode='wb')
    waveobj.setnchannels(n_channels)
    waveobj.setframerate(sr)
    waveobj.setsampwidth(2)
    waveobj.setcomptype('NONE', 'NONE')
    waveobj.writeframes(b''.join([struct.pack('<h', x) for x in data_audio]))
    val = fp.getvalue()
    waveobj.close()
    data_audio = base64 = base64.b64encode(val).decode('ascii')
    src = """data:{type};base64,{base64}""".format(type="audio/wav",
                                                   base64=data_audio)
    return src
