from numpy import empty
from fastexp import fastexp as _fastexp

def fastexp(x, out=None):
    if out is None:
        y = empty(x.shape)
        _fastexp(x,y)
        return y
    _fastexp(x, out)
    return out
