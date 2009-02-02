import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import numpy, pylab, time

N = 1000
block = (512,1,1)
grid = (int(N/512)+1,1)

mod = drv.SourceModule("""
__

__global__ void stateupdate(double *v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  v[i] += 0.0001*v__tmp;
}
""")

stateupdate = mod.get_function("stateupdate")

v = gpuarray.to_gpu(numpy.randn(N))

stateupdate(v, w, block=block, grid=grid)
