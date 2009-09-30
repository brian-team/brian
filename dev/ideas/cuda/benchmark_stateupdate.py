import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import bisect
import numpy, pylab, time

N = 1000000
blocksize = 512
duration = 10000

N = int(numpy.ceil(1.*N/blocksize)*blocksize)

if drv.get_version()==(2,0,0): # cuda version
    precision = 'float'
elif drv.get_version()>(2,0,0):
    precision = 'double'
else:
    raise Exception,"CUDA 2.0 required"

if precision=='double':
    mydtype = numpy.float64
else:
    mydtype = numpy.float32

block = (blocksize,1,1)
grid = (N/blocksize,1)

mod = drv.SourceModule("""
/*__global__ void stateupdate(SCALAR *V_arr, SCALAR *ge_arr, SCALAR *gi_arr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    SCALAR V = V_arr[i];
    SCALAR ge = ge_arr[i];
    SCALAR gi = gi_arr[i]; 
//    SCALAR V__tmp = (ge+gi-(V+0.049))/0.02;
//    SCALAR ge__tmp = -ge/0.005;
//    SCALAR gi__tmp = -gi/0.01;
    SCALAR V__tmp = (ge+gi-(V+0.049))*50;
    SCALAR ge__tmp = -ge*200;
    SCALAR gi__tmp = -gi*100;
    V_arr[i] = V+0.0001*V__tmp;
    ge_arr[i] = ge+0.0001*ge__tmp;
    gi_arr[i] = gi+0.0001*gi__tmp;
}*/

__global__ void stateupdate(SCALAR *V_arr, SCALAR *ge_arr, SCALAR *gi_arr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    SCALAR V = V_arr[i];
    SCALAR ge = ge_arr[i];
    SCALAR gi = gi_arr[i];
    V_arr[i] = V+1;
    ge_arr[i] = ge+1;
    gi_arr[i] = gi+1;
}

__global__ void stateupdate_noglobal(SCALAR *V_arr, SCALAR *ge_arr, SCALAR *gi_arr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    SCALAR V = 0.;//V_arr[i];
    SCALAR ge = 0.;//ge_arr[i];
    SCALAR gi = 0.;//gi_arr[i]; 
    SCALAR V__tmp = (ge+gi-(V+0.049))/0.02;
    SCALAR ge__tmp = -ge/0.005;
    SCALAR gi__tmp = -gi/0.01;
    V = V+0.0001*V__tmp;
    ge = ge+0.0001*ge__tmp;
    gi = gi+0.0001*gi__tmp;
}
""".replace('SCALAR',precision))
stateupdate = mod.get_function("stateupdate")
stateupdate_noglobal = mod.get_function("stateupdate_noglobal")

V = gpuarray.to_gpu(numpy.array(numpy.random.rand(N)*0.01-0.06,dtype=mydtype))
ge = gpuarray.to_gpu(numpy.zeros(N,dtype=mydtype))
gi = gpuarray.to_gpu(numpy.zeros(N,dtype=mydtype))

stateupdate.prepare(('i', 'i', 'i'), block)
stateupdate_args = (grid, int(V.gpudata), int(ge.gpudata), int(gi.gpudata))
stateupdate.prepared_call(*stateupdate_args)
#stateupdate_noglobal.prepared_call(*stateupdate_args)

def run_sim(stateupdate=stateupdate):
    for t in xrange(duration):
        stateupdate.launch_grid(*grid)

start = time.time()
run_sim()
duration_stateupdate = time.time()-start

start = time.time()
run_sim(stateupdate_noglobal)
duration_stateupdate_noglobal = time.time()-start

data_copied_gb = N*mydtype().nbytes*3*duration*2./1024.**3
transfer_rate_gb_sec = data_copied_gb/(duration_stateupdate-duration_stateupdate_noglobal)

print 'N:', N, 'blocksize:', blocksize, 'numsteps:', duration
print 'GPU time:', duration_stateupdate
print 'GPU time (no global memory access):', duration_stateupdate_noglobal
print 'Data transferred (GB):', data_copied_gb
print 'Data transfer rate (GB/sec):', transfer_rate_gb_sec
