from numpy.random import *
from pycuda.autoinit import *
from pycuda import *
#from pycuda.gpuarray import *
from pycuda import driver as drv
from ctypes import *
from numpy import *

lib=cdll.LoadLibrary('testchagpp')
find_positive=lib.find_positive
print find_positive

x=randn(100)
I=(x>0).nonzero()[0]

y=drv.mem_alloc(100*8)
drv.memcpy_htod(y, x)

out=drv.mem_alloc(100*8)
count=drv.mem_alloc(4)

#y = to_gpu(x)
#out = to_gpu(zeros(len(x)))
#count = to_gpu(array([0], dtype=int))

find_positive(int(y), int(y)+len(x)*8, int(out), int(count))

#find_positive(int(y.gpudata), int(y.gpudata)+len(x)*8,
#              int(out.gpudata), int(count.gpudata))

z=zeros(100)
drv.memcpy_dtoh(z, out)

c=array([0], dtype=int)
drv.memcpy_dtoh(c, count)

z=z[:c[0]]

#z = out.get()[:count.get()[0]]

print z-x[I]
