from numpy import *

__all__ = ['DynamicArray']

def getslices(shape):
    return tuple(slice(0, x) for x in shape)

class DynamicArray(object):
    '''
    An N-dimensional dynamic array class
    
    The array can be resized in any dimension, and the class will handle
    allocating a new block of data and copying when necessary.
    
    .. warning::
        The data will NOT be contiguous for >1D arrays. To ensure this, you will
        either need to use 1D arrays, or to copy the data, or use the shrink
        method with the current size (although note that in both cases you
        negate the memory and efficiency benefits of the dynamic array).
    
    Initialisation arguments:
    
    ``shape``, ``dtype``
        The shape and dtype of the array to initialise, as in Numpy. For 1D
        arrays, shape can be a single int, for ND arrays it should be a tuple.
    ``factor``
        The resizing factor (see notes below). Larger values tend to lead to
        more wasted memory, but more computationally efficient code.
        
    The array is initialised with zeros. The data is stored in the attribute
    ``data`` which is a Numpy array.
    
    **Methods**
    
    .. automethod:: resize
    .. automethod:: shrink
    
    Some numpy methods are implemented and can work directly on the array object,
    including ``len(arr)``, ``arr[...]`` and ``arr[...]=...``. In other cases,
    use the ``data`` attribute.
    
    **Usage example**
    
    ::

        x = DynamicArray((2, 3), dtype=int)
        x[:] = 1
        x.resize((3, 3))
        x[:] += 1
        x.resize((3, 4))
        x[:] += 1
        x.resize((4, 4))
        x[:] += 1
        x.data[:] = x.data**2
        print x.data
        
    This should give the output::

        [[16 16 16  4]
         [16 16 16  4]
         [ 9  9  9  4]
         [ 1  1  1  1]]    
    
    **Notes**
    
    The dynamic array returns a ``data`` attribute which is a view on the larger
    ``_data`` attribute. When a resize operation is performed, and a specific
    dimension is enlarged beyond the size in the ``_data`` attribute, the size
    is increased to the larger of ``cursize*factor`` and ``newsize``. This
    ensures that the amortized cost of increasing the size of the array is O(1).  
    '''
    def __init__(self, shape, dtype=float, factor=2):
        if isinstance(shape, int):
            shape = (shape,)
        self._data = zeros(shape, dtype=dtype)
        self.data = self._data
        self.dtype = dtype
        self.shape = self._data.shape
        self.factor = factor
    
    def resize(self, newshape):
        '''
        Resizes the data to the new shape, which can be a different size to the
        current data, but should have the same rank, i.e. same number of
        dimensions.
        '''
        if isinstance(newshape, int):
            newshape = (newshape,)
        datashapearr = array(self._data.shape)
        shapearr = array(self.shape)
        newshapearr = array(newshape)
        if (shapearr==newshapearr).all():
            return
        resizedimensions = newshapearr>datashapearr
        if resizedimensions.any():
            # resize of the data is needed
            minnewshapearr = datashapearr.copy()
            dimstoinc = minnewshapearr[resizedimensions]
            incdims = array(dimstoinc*self.factor, dtype=int)
            newdims = maximum(incdims, dimstoinc+1)
            minnewshapearr[resizedimensions] = newdims
            newshapearr = maximum(newshapearr, minnewshapearr)
            newdata = zeros(tuple(newshapearr), dtype=self.dtype)
            slices = getslices(self._data.shape)
            newdata[slices] = self._data
            self._data = newdata
        self.data = self._data[getslices(newshape)]
        self.shape = self.data.shape
        
    def shrink(self, newshape):
        '''
        Reduces the data to the given shape, which should be smaller than the
        current shape. :meth:`resize` can also be used with smaller values, but
        it will not shrink the allocated memory, whereas :meth:`shrink` will
        reallocate the memory. This method should only be used infrequently, as
        if it is used frequently it will negate the computational efficiency
        benefits of the DynamicArray.
        '''
        if isinstance(newshape, int):
            newshape = (newshape,)
        shapearr = array(self.shape)
        newshapearr = array(newshape)
        if (newshapearr<=shapearr).all():
            newdata = zeros(newshapearr, dtype=self.dtype)
            newdata[:] = self._data[getslices(newshapearr)]
            self._data = newdata
            self.shape = tuple(newshapearr)
            self.data = self._data
    
    def __getitem__(self, item):
        return self.data.__getitem__(item)
    
    def __getslice__(self, start, end):
        return self.data.__getslice__(start, end)
    
    def __setitem__(self, item, val):
        self.data.__setitem__(item, val)
        
    def __setslice__(self, start, end, val):
        self.data.__setslice__(start, end, val)
        
    def __len__(self):
        return len(self.data)
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()
    
            
if __name__=='__main__':
    if 1:
        x = DynamicArray(3, dtype=int)
        x[:] = [1, 2, 3]
        print x
        x.resize(5)
        print x
        x.shrink(4)
        print x
    if 1:
        x = DynamicArray((2, 3), dtype=int)
        x[:] = 1
        x.resize((3, 3))
        x[:] += 1
        x.resize((3, 4))
        x[:] += 1
        x.resize((4, 4))
        x[:] += 1
        x.data[:] = x.data**2
        print x.data        
    if 1:
        def doprint():
            print x.data.shape, x._data.shape
            print x.data
            print x._data
            print
        x = DynamicArray((2, 3))
        x[:] = 1
        doprint()
        x.resize((2, 3))
        doprint()
        x.resize((3, 3))
        x[:] += 1
        doprint()
        x.resize((3, 4))
        x[:] += 1
        doprint()
        x.resize((4, 4))
        x[:] += 1
        doprint()
        x.resize((9, 7))
        x[:] += 1
        doprint()
        x.resize((4, 4))
        x[:] += 1
        doprint()
        x.shrink((4, 2))
        x[:] += 1
        doprint()
        