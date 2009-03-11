import numpy

class TimedArray(numpy.ndarray):
    def __new__(subtype, arr, times=None, dt=None):
        return numpy.array(arr, copy=False).view(subtype)
    def __init__(self, arr, times=None, dt=None):
        if (times is None and dt is None) or (times is not None and dt is not None):
            raise ArgumentError