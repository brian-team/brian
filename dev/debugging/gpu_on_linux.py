from brian.hears import *
import brian.hears.filtering as filtering
from numpy import ones, int32
import multiprocessing

import pycuda
#import pycuda.autoinit as autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule



import os, sys
import atexit

#def set_gpu_device(n):
#    """
#    This function makes pycuda use GPU number n in the system.
#    """
#    try:
#        pycuda.context.detach()
#    except:
#        pass
#    pycuda.context = drv.Device(n).make_context()
#
#def close_cuda():
#    """
#    Closes the current context. MUST be called at the end of the script.
#    """
#    if pycuda.context is not None:
#        try:
#            pycuda.context.pop()
#            pycuda.context = None
#        except:
#            pass
#
#atexit.register(close_cuda)

def doit(x):

   # drv.init()
#    print id(pycuda.context)

    filtering.set_gpu_device(x)
#    print 'a'
#    pycuda.context = drv.Device(0).make_context()
#    print 'b'
    sys.stdin = file(os.devnull)
    sys.stdout = file(os.devnull)

    code = '''
    __global__ void test(double *x, int n)
    {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i>=n) return;
     x[i] *= 2.0;
    }
    '''

    mod = SourceModule(code)
    f = mod.get_function('test')
    x = gpuarray.to_gpu(ones(100))
    f(x, int32(100), block=(100, 1, 1))
    y = x.get()
    
    filtering.close_cuda()
    
    return y

if __name__ == '__main__':
#    pycuda.context=None
    #import sys
#    print 'A'
#    set_gpu_device(0)
#    print 'B'
#    filtering.set_gpu_device(0)
#    print 'C'
#    filtering.set_gpu_device(0)
#    print 'D'

    #drv.init()
    #drv.init()

    #doit(0)
    #print id(pycuda.context)
    #close_cuda()

    pool = multiprocessing.Pool(2)
    result = pool.map(doit, [0,0])
    print result
