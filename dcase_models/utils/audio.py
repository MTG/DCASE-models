import struct
import wave
from io import BytesIO
from librosa import db_to_power
from librosa.util import nnls
from librosa.core import griffinlim
import numpy as np


# path, sr, scaler, mel_basis, audio_hop, audio_win
def mel_spec_to_audio(melspec_db, mel_basis, audio_hop, audio_win):
    """
    Generate self.audios with the data of self.mel_spectrograms and the
    parameters given by self.convert_audio_params.
    """
    melspec = db_to_power(melspec_db)
    inverse = nnls(mel_basis, melspec)
    spec = np.power(inverse, 1./2.0, out=inverse)

    audio = griffinlim(spec, hop_length=audio_hop,
                       win_length=audio_win,
                       center=False)

    audio_save = audio[200:-200]

    return audio_save


def encode_audio(data, sr):
    import base64
    # print(data.shape,len(data.shape))
    if len(data.shape) == 1:
        n_channels = 1
        data_audio = data
    else:
        n_channels = data.shape[1]
        data_audio = data.ravel()

    data_audio = data_audio/np.amax(data_audio)
    data_audio = np.int16(data_audio * 32767).tolist()
    # print(len(data_audio))

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
