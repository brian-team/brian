from numpy.random import *
from pycuda.autoinit import *
from pycuda import *
from pycuda.gpuarray import *
from ctypes import *
from numpy import *

lib = cdll.LoadLibrary('testchagpp')
find_positive = lib.find_positive
print find_positive

x = randn(100)
I = (x>0).nonzero()[0]

y = to_gpu(x)
out = to_gpu(zeros(len(x)))
count = to_gpu(array([0], dtype=int))

find_positive(int(y.gpudata), int(y.gpudata)+len(x)*8,
              int(out.gpudata), int(count.gpudata))

z = out.get()[:count.get()[0]]

print z-x[I]
