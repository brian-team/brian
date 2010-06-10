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

def doit(x):

    print id(pycuda.context)

    filtering.set_gpu_device(x)

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

    #import sys
#    print 'A'
#    filtering.set_gpu_device(0)
#    print 'B'
#    filtering.set_gpu_device(0)
#    print 'C'
#    filtering.set_gpu_device(0)
#    print 'D'

   # doit(0)
    print id(pycuda.context)
    pool = multiprocessing.Pool(1)
    result = pool.map(doit, [0])
    #print result
