import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import bisect
import numpy, pylab, time
from itertools import repeat
import cProfile as profile
import pstats
import brian

N = 10000
blocksize = 512
N = int(numpy.ceil(1. * N / blocksize) * blocksize)
duration = 10000
record = False
doprofile = False
sparseness = 80. / N

if drv.get_version() == (2, 0, 0): # cuda version
    precision = 'float'
elif drv.get_version() > (2, 0, 0):
    precision = 'double'
else:
    raise Exception, "CUDA 2.0 required"

if precision == 'double':
    mydtype = numpy.float64
else:
    mydtype = numpy.float32

block = (blocksize, 1, 1)
grid = (N / blocksize, 1)
Ne = int(N * 0.8)
Ni = N - Ne

shared_memory_size = 2 ** 14 # 16k
matrix_split_size = blocksize * (shared_memory_size // mydtype().nbytes // blocksize)
print 'Matrix split size:', matrix_split_size

mod = SourceModule("""
__global__ void stateupdate_and_threshold(SCALAR *V_arr, SCALAR *ge_arr, SCALAR *gi_arr, int *spikes, bool *spiked, unsigned int *global_j, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    SCALAR V = V_arr[i];
    SCALAR ge = ge_arr[i];
    SCALAR gi = gi_arr[i]; 
    SCALAR V__tmp = (ge+gi-(V+0.049))/0.02;
    SCALAR ge__tmp = -ge/0.005;
    SCALAR gi__tmp = -gi/0.01;
    V += 0.0001*V__tmp;
    ge += 0.0001*ge__tmp;
    gi += 0.0001*gi__tmp;
    V_arr[i] = V;
    ge_arr[i] = ge;
    gi_arr[i] = gi;
    bool this_spiked = V>-0.05; 
    spiked[i] = this_spiked;
    if(this_spiked)
        spikes[atomicInc(global_j, N)] = i;
}

__global__ void reset(SCALAR *V, bool *spiked)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool has_spiked = spiked[i];
    //V[i] = (V[i]*!has_spiked)+(-0.06)*has_spiked; // i.e. V[i]=-0.06 if spiked[i] - but this operation requires two global mem accesses
    if(has_spiked)
      V[i] = -0.06;
}
""".replace('SCALAR', precision))
stateupdate_and_threshold = mod.get_function("stateupdate_and_threshold")
reset = mod.get_function("reset")

V = gpuarray.to_gpu(numpy.array(numpy.random.rand(N) * 0.01 - 0.06, dtype=mydtype))
ge = gpuarray.to_gpu(numpy.zeros(N, dtype=mydtype))
gi = gpuarray.to_gpu(numpy.zeros(N, dtype=mydtype))

we = 0.00162
wi = -0.009
P = brian.NeuronGroup(N, model='V:1')
Pe = P.subgroup(Ne)
Pi = P.subgroup(Ni)
Ce = brian.Connection(Pe, P, sparseness=sparseness, weight=we).W.connection_matrix()
Ci = brian.Connection(Pi, P, sparseness=sparseness, weight=wi).W.connection_matrix()
Ce.alldata = numpy.array(Ce.alldata, dtype=mydtype)
Ci.alldata = numpy.array(Ci.alldata, dtype=mydtype)
Ce_alldata = drv.mem_alloc(Ce.alldata.nbytes)
drv.memcpy_htod(Ce_alldata, Ce.alldata)
Ce_allj = drv.mem_alloc(Ce.allj.nbytes)
drv.memcpy_htod(Ce_allj, Ce.allj)
Ce_rowind = drv.mem_alloc(Ce.rowind.nbytes)
drv.memcpy_htod(Ce_rowind, Ce.rowind)
Ci_alldata = drv.mem_alloc(Ci.alldata.nbytes)
drv.memcpy_htod(Ci_alldata, Ci.alldata)
Ci_allj = drv.mem_alloc(Ci.allj.nbytes)
drv.memcpy_htod(Ci_allj, Ci.allj)
Ci_rowind = drv.mem_alloc(Ci.rowind.nbytes)
drv.memcpy_htod(Ci_rowind, Ci.rowind)
