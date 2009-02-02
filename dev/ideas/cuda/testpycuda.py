import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import numpy, pylab, time

N = 10000
duration = int(0.1*10000)
record = False
showfinal = True
doget = True
dosync = False
block = (512,1,1)
grid = (int(N/512)+1,1)

#Device attributes
#MAX_THREADS_PER_BLOCK : 512
#MAX_BLOCK_DIM_X : 512
#MAX_BLOCK_DIM_Y : 512
#MAX_BLOCK_DIM_Z : 64
#MAX_GRID_DIM_X : 65535
#MAX_GRID_DIM_Y : 65535
#MAX_GRID_DIM_Z : 1
#MAX_SHARED_MEMORY_PER_BLOCK : 16384
#TOTAL_CONSTANT_MEMORY : 65536
#WARP_SIZE : 32
#MAX_PITCH : 262144
#MAX_REGISTERS_PER_BLOCK : 16384
#CLOCK_RATE : 1296000
#TEXTURE_ALIGNMENT : 256
#GPU_OVERLAP : 1
#MULTIPROCESSOR_COUNT : 30

mod = drv.SourceModule("""
__global__ void stateupdate(double *v, double *w)
{
  //const int i = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double v__tmp = w[i]*w[i]/(100*0.001);
  double w__tmp = -v[i]/(100*0.001);
  v[i] += 0.0001*v__tmp;
  w[i] += 0.0001*w__tmp;
}
""")

stateupdate = mod.get_function("stateupdate")

v = gpuarray.to_gpu(numpy.ones(N))
w = gpuarray.to_gpu(numpy.zeros(N))

#ov = numpy.zeros(N)
ov = drv.pagelocked_zeros((N,), dtype=float)

recv = []
start = time.time()
for _ in xrange(duration):
    stateupdate(v, w, block=block, grid=grid)
    if record:
        recv.append(v.get())
    elif doget:
        v.get(ov)
    elif dosync:
        autoinit.context.synchronize()
print 'GPU code:', time.time()-start

if record:
    x = numpy.array(recv)
    pylab.plot(x[:,0])
if showfinal:
    pylab.figure()
    pylab.plot(v.get())
if record or showfinal:
    pylab.show()
    
'''
Some results:

dV/dt = W*W/(100*ms) : 1
dW/dt = -V/(100*ms) : 1

10k neurons for 1000 time steps
Python: 0.5s
C++: 0.4s
GPUgetpagelocked: 0.1s 
GPUsync: 0.047s

100k neurons
Python: 9.3s
C++: 2.9s
GPUgetpagelocked: 0.45s
GPUsync: 0.16s

1M neurons
Python: 111s
C++: 27.6s
GPUgetpagelocked: 4.1s
GPUsync: 1.2s
'''