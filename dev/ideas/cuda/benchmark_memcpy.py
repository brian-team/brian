from numpy import *
import pycuda
import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import time

N = 100000000
steps = 10
use_pagelocked = False
do_get = True
do_set = False

getset_factor = float(int(do_get)+int(do_set))

a_cpu = zeros(N)
a_gpu = pycuda.gpuarray.to_gpu(a_cpu)

start = time.time()

for _ in xrange(steps):
    if do_get: a_gpu.set(a_cpu)
    if do_set: a_gpu.get(a_cpu)
    
duration = time.time()-start

print 'Array size', N
print 'Pagelocked', use_pagelocked
print 'Number of steps', steps
print 'Download', do_get, 'upload', do_set
print 'Total memory copied (GB)', N*8.*steps*getset_factor/1024.**3
print 'Time taken (sec)', duration
print 'Transfer rate (GB/sec)', N*8.*steps*getset_factor/1024.**3/duration