from numpy import empty, ndarray, exp
from fastexp import fastexp as _fastexp

def fastexp(x, out=None):
    if not isinstance(x, ndarray) or len(x.shape) > 1:
        return exp(x)
    if out is None:
        y = empty(x.shape)
        _fastexp(x, y)
        return y
    _fastexp(x, out)
    return out
