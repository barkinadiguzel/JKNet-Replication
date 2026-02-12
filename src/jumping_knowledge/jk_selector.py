from .jk_concat import JKConcat
from .jk_maxpool import JKMaxPool
from .jk_lstm_attention import JKLSTM

def get_jk(mode, dim=None):
    if mode == "concat":
        return JKConcat()
    elif mode == "maxpool":
        return JKMaxPool()
    elif mode == "lstm":
        return JKLSTM(dim)
    else:
        raise ValueError("Unknown JK mode")
