from numpy import zeros
import numpy
from pycuda import autoinit
from pycuda import gpuarray
from pycuda import compiler
x = gpuarray.to_gpu(zeros(10, numpy.float32))
src = '''
__global__ void f(float *x)
{
 x[0] = exp(x[0]);
}
'''
gpu_mod = compiler.SourceModule(src)
gpu_func = gpu_mod.get_function("f")
gpu_func(x, block=(1, 1, 1), grid=(1, 1))
print x.get()
