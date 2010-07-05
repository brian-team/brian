'''
GPU Buffering class

See docstring for GPUBufferedArray for details.
'''
from brian import *
import pycuda
import pycuda.gpuarray
import numpy
from numpy import intp

__all__ = ['SynchronisationError', 'GPUBufferedArray']

if __name__ == '__main__':
    DEBUG_BUFFER_CACHE = True
else:
    DEBUG_BUFFER_CACHE = False

numpy_inplace_methods = set([
     '__iadd__',
     '__iand__',
     '__idiv__',
     '__ifloordiv__',
     '__ilshift__',
     '__imod__',
     '__imul__',
     '__ior__',
     '__ipow__',
     '__irshift__',
     '__isub__',
     '__itruediv__',
     '__ixor__',
     '__setitem__',
     '__setslice__',
     'byteswap',
     'fill',
     'put',
     'sort',
     ])

numpy_access_methods = set([
     '__abs__',
     '__add__',
     '__and__',
     '__array__',
     '__contains__',
     '__copy__',
     '__deepcopy__',
     '__delattr__',
     '__delitem__',
     '__delslice__',
     '__div__',
     '__divmod__',
     '__eq__',
     '__float__',
     '__floordiv__',
     '__ge__',
     '__getitem__',
     '__getslice__',
     '__gt__',
     '__hash__',
     '__hex__',
     '__iadd__',
     '__iand__',
     '__idiv__',
     '__ifloordiv__',
     '__ilshift__',
     '__imod__',
     '__imul__',
     '__index__',
     '__int__',
     '__invert__',
     '__ior__',
     '__ipow__',
     '__irshift__',
     '__isub__',
     '__iter__',
     '__itruediv__',
     '__ixor__',
     '__le__',
     '__long__',
     '__lshift__',
     '__lt__',
     '__mod__',
     '__mul__',
     '__ne__',
     '__neg__',
     '__nonzero__',
     '__oct__',
     '__or__',
     '__pos__',
     '__pow__',
     '__radd__',
     '__rand__',
     '__rdiv__',
     '__rdivmod__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rfloordiv__',
     '__rlshift__',
     '__rmod__',
     '__rmul__',
     '__ror__',
     '__rpow__',
     '__rrshift__',
     '__rshift__',
     '__rsub__',
     '__rtruediv__',
     '__rxor__',
     '__setitem__',
     '__setslice__',
     '__setstate__',
     '__str__',
     '__sub__',
     '__truediv__',
     '__xor__',
     'all',
     'any',
     'argmax',
     'argmin',
     'argsort',
     'astype',
     'byteswap',
     'choose',
     'clip',
     'compress',
     'conj',
     'conjugate',
     'copy',
     'cumprod',
     'cumsum',
     'diagonal',
     'dump',
     'dumps',
     'fill',
     'flatten',
     'getfield',
     'item',
     'itemset',
     'max',
     'mean',
     'min',
     'nonzero',
     'prod',
     'ptp',
     'put',
     'ravel',
     'repeat',
     'reshape',
     'resize',
     'round',
     'searchsorted',
     'setfield',
     'setflags',
     'sort',
     'squeeze',
     'std',
     'sum',
     'swapaxes',
     'take',
     'tofile',
     'tolist',
     'tostring',
     'trace',
     'transpose',
     'var',
     'view']) - numpy_inplace_methods


class SynchronisationError(RuntimeError):
    '''
    Indicates that the GPU and CPU data are in conflict (i.e. both have changed)
    '''
    pass

# These two functions are decorators to be applied to the numpy
# methods. The inplace version sets the cpu data changed flag
# after it has run, the access version only makes sure that
# the data has been synchronised to the CPU.
def gpu_buffered_array_inplace_method(func):
    def new_func(self, *args, **kwds):
        self.sync_to_cpu()
        rval = func(self, *args, **kwds)
        self._cpu_data_changed = True
        return rval
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    return new_func

def gpu_buffered_array_access_method(func):
    def new_func(self, *args, **kwds):
        self.sync_to_cpu()
        rval = func(self, *args, **kwds)
        return rval
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    return new_func


class DecoupledGPUBufferedArray(numpy.ndarray):
    pass


class GPUBufferedArray(numpy.ndarray):
    '''
    An object that automatically manages a numpy array that is stored on the CPU and GPU
    
    The idea is that whenever data is accessed on either the GPU or CPU, this object
    performs any synchronisation operations. So for example if you change a value in the
    array on the CPU and then run a function on the GPU, the data from the CPU side
    needs to be copied to the GPU before calling the GPU function.
    
    The best practice is to use the ``cpu_array`` and ``gpu_array`` or ``gpu_dev_alloc``
    attributes explicitly, and call the ``changed_cpu_data()`` and ``changed_gpu_data()`` methods
    after modifying data. However, the class does its best to be intelligent about
    when changes to data have happened so these arrays can be used in code that is
    not GPU aware to some extent. See implementation notes below for details on when
    this will fail.
    
    **Initialisation**
    
    Initialising with a numpy array creates a ``pycuda.gpuarray.GPUArray`` instance
    to store the data on the GPU. If you want, you can specify a pre-existing
    ``GPUArray`` or ``pycuda.DeviceAllocation`` instance. Initialising with a
    pagelocked numpy array will make copies faster.
    
    **Attributes**
    
    Each of these attributes returns a reference to data on the CPU or GPU (and
    like everything in this class, will synchronise if necessary). Obtaining a
    reference to CPU/GPU data is taken as implying that the data will be
    changed by subsequent code, and so the CPU/GPU data is marked as changed.
    To obtain references without marking the data as changed, use the
    attribute with ``_nomodify`` appended. If you are obtaining and storing
    a reference, you need to explicitly mark it as modified when you make
    a change (see the ``changed_gpu_data()`` and ``changed_cpu_data()`` methods below).
    
    ``cpu_array``, ``cpu_array_nomodify``
        A reference to the array on the CPU.
    ``gpu_array``, ``gpu_array_nomodify``
        If the GPU data is a ``GPUArray``, this will return it, otherwise it will
        throw an error.
    ``gpu_dev_alloc``, ``gpu_dev_alloc_nomodify``
        Return the ``DeviceAllocation`` for the GPU data.
    ``gpu_pointer``, ``gpu_pointer_nomodify``
        Returns an integer pointer to the GPU data.
    
    **Methods**
    
    ``changed_gpu_data()``, ``changed_cpu_data()``
        Mark the data on the GPU or CPU as modified. You need to do this
        explicitly if you obtained and stored a reference to the CPU or GPU
        data.
    ``sync_to_gpu()``, ``sync_to_cpu()``
        Explicitly make sure that the data on the CPU/GPU has been copied to
        the GPU/CPU respectively if they are marked as changed.
    ``sync()``
        Explicitly synchronise the CPU and GPU.
    
    **Implementation**
    
    The implementation is not clever about copying data, it either copies all the data
    or none at all. Internally it stores two flags to indicate whether data on the CPU
    or GPU has changed, which it tries its best to be accurate but cannot guarantee
    (see below), and performs a copy when needed (subsequently updating these flags).
    
    Results of numpy functions, except for a handful of explicitly inplace operators,
    do not mark the data as changed. So doing, for example, ``clip(y,0,1,out=y)``
    will not mark the data in ``y`` as changed.
    '''
    def __new__(subtype, cpu_arr, gpu_arr=None):
        return numpy.array(cpu_arr, copy=False).view(subtype)

    def __init__(self, cpu_arr, gpu_arr=None):
        if gpu_arr is None:
            if DEBUG_BUFFER_CACHE:
                print 'Initialising and copying GPU memory'
            self._gpu_arr = pycuda.gpuarray.to_gpu(self)
            self._gpu_data_changed = False
            self._cpu_data_changed = False
        else:
            self._gpu_arr = gpu_arr
            self._gpu_data_changed = False
            self._cpu_data_changed = True

    def _check_synchronisation(self):
        if not hasattr(self, '_cpu_data_changed'):
            # This happens if you do, for example, y[0,:][:]=1 because the y[0,:] will
            # return an uninitialised GPUBufferedArray. At the moment, we don't
            # handle this case (although perhaps in the case of slicing we could?) so
            # maybe we should throw an error here? The alternative is to try to minimise
            # the impact on code (although it will have decoupled the array from the
            # GPU data, which may not be expected).
            self.__class__ = DecoupledGPUBufferedArray
            self._cpu_data_changed = False
            self._gpu_data_changed = False
            return
        if self._cpu_data_changed and self._gpu_data_changed:
            raise SynchronisationError('GPU and CPU data desynchronised.')

    def sync(self):
        self.sync_to_gpu()
        self.sync_to_cpu()

    def sync_to_gpu(self):
        self._check_synchronisation()
        if self._cpu_data_changed:
            gpu_arr = self._gpu_arr
            if isinstance(gpu_arr, pycuda.gpuarray.GPUArray):
                gpu_arr.set(self)
            else:
                pycuda.driver.memcpy_htod(gpu_arr, self)
            self._cpu_data_changed = False
            if DEBUG_BUFFER_CACHE:
                print 'Synchronised to GPU'

    def sync_to_cpu(self):
        self._check_synchronisation()
        if self._gpu_data_changed:
            gpu_arr = self._gpu_arr
            if isinstance(gpu_arr, pycuda.gpuarray.GPUArray):
                gpu_arr.get(self)
            else:
                pycuda.driver.memcpy_dtoh(self, gpu_arr)
            self._gpu_data_changed = False
            if DEBUG_BUFFER_CACHE:
                print 'Synchronised to CPU'

    def changed_gpu_data(self):
        self._gpu_data_changed = True
        self._check_synchronisation()

    def changed_cpu_data(self):
        self._cpu_data_changed = True
        self._check_synchronisation()

    def get_cpu_array(self, modify=True):
        self.sync_to_cpu()
        if modify: # assume that the user is going to modify the data
            self._cpu_data_changed = True
        return self

    def get_gpu_array(self, modify=True):
        self.sync_to_gpu()
        if modify: # assume that the user is going to modify the data
            self._gpu_data_changed = True
        if isinstance(self._gpu_arr, pycuda.gpuarray.GPUArray):
            return self._gpu_arr
        raise TypeError('GPU buffer is not a GPUArray')

    def get_gpu_dev_alloc(self, modify=True):
        self.sync_to_gpu()
        if modify: # assume that the user is going to modify the data
            self._gpu_data_changed = True
        if isinstance(self._gpu_arr, pycuda.gpuarray.GPUArray):
            return self._gpu_arr.gpudata
        elif isinstance(self._gpu_arr, pycuda.driver.DeviceAllocation):
            return self._gpu_arr
        raise TypeError('gpu_arr should be a DeviceAllocation or GPUArray.')

    def get_gpu_pointer(self):
        return intp(self.gpu_dev_alloc)

    cpu_array = property(fget=get_cpu_array)
    gpu_array = property(fget=get_gpu_array)
    gpu_dev_alloc = property(fget=get_gpu_dev_alloc)
    gpu_pointer = property(fget=get_gpu_pointer)
    cpu_array_nomodify = property(fget=lambda self:self.get_cpu_array(False))
    gpu_array_nomodify = property(fget=lambda self:self.get_gpu_array(False))
    gpu_dev_alloc_nomodify = property(fget=lambda self:self.get_gpu_dev_alloc(False))
    gpu_pointer_nomodify = property(fget=lambda self:self.get_gpu_pointer(False))
    for __name in numpy_inplace_methods:
        exec __name + ' = gpu_buffered_array_inplace_method(numpy.ndarray.' + __name + ')'
    for __name in numpy_access_methods:
        exec __name + ' = gpu_buffered_array_access_method(numpy.ndarray.' + __name + ')'

    def __array_wrap__(self, obj, context=None):
        # normally, __array_wrap__ calls __array_finalize__ to convert the result obj into an object of its own type
        # but here, we don't want that, we want it to be a straight numpy array which doesn't maintain a connection
        # to the GPU array.
        return obj

if __name__ == '__main__':
    import pycuda.autoinit
    from pycuda import driver as drv
    def gpuarrstatus(z):
        return '(cpu_changed=' + str(z._cpu_data_changed) + ', gpu_changed=' + str(z._gpu_data_changed) + ')'
    x = array([1, 2, 3, 4], dtype=numpy.float32)
    print '+ About to initialise GPUBufferedArray'
    y = GPUBufferedArray(x)
    print '- Initialised GPUBufferedArray', gpuarrstatus(y)
    print '+ About to set item on CPU array', gpuarrstatus(y)
    y[3] = 5
    print '- Set item on CPU array', gpuarrstatus(y)
    mod = drv.SourceModule('''
    __global__ void doubleit(float *y)
    {
        int i = threadIdx.x;
        y[i] *= 2.0;
    }
    ''')
    doubleit = mod.get_function("doubleit")
    print '+ About to call GPU function', gpuarrstatus(y)
    doubleit(y.gpu_array, block=(4, 1, 1))
    print '- Called GPU function', gpuarrstatus(y)
    print '+ About to print CPU array', gpuarrstatus(y)
    print y
    print '- Printed CPU array', gpuarrstatus(y)
