from numpy import *

class DynamicArray(object):
    def __init__(self, shape, dtype):
        self._data = zeros(shape, dtype=dtype)
        self.shape = self._data.shape
    def resize(self, newshape):
        datashapearr = array(self._data.shape)
        shapearr = array(self.shape)
        newshapearr = array(newshape)
        resizedimensions = newshapearr>datashapearr
        if resizedimensions.any():
            # resize of the data is needed
            minnewshapearr = datashapearr.copy()
            minnewshapearr[resizedimensions] *= 2
            newshapearr = maximum(newshapearr, minnewshapearr)
            newdata = empty