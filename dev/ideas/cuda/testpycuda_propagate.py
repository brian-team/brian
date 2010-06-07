import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import numpy, pylab, time

N=512*8
spikes=[i for i in range(N) if numpy.random.rand()<0.1]
blocksize=512
block=(blocksize, 1, 1)
grid=(N/blocksize, 1)
repeats=1000

print 'N:', N
print 'numspikes:', len(spikes), 'proportion', float(len(spikes))/N
print 'repeats:', repeats
print 'block:', block, 'grid:', grid

mod=drv.SourceModule("""
__global__ void propagate(int *spikes, int numspikes, float *v, float *W, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int j=0; j<numspikes; j++)
        v[i] += W[i+N*spikes[j]];
}
""")

propagate=mod.get_function("propagate")

W=numpy.random.randn(N, N)
W_gpu=gpuarray.to_gpu(numpy.array(W, dtype=numpy.float32))

v=gpuarray.to_gpu(numpy.zeros(N, dtype=numpy.float32))
v_pre=v.get()

gpu_spikes=drv.mem_alloc(4*len(spikes))
spikes=numpy.array(spikes, dtype=int)
drv.memcpy_htod(gpu_spikes, spikes)

start=time.time()
for _ in xrange(repeats):
    propagate(gpu_spikes, numpy.int32(len(spikes)), v, W_gpu, numpy.int32(N),
              block=block, grid=grid)
autoinit.context.synchronize()
print 'GPU propagation:', time.time()-start
v_post_gpu=v.get()
gputime=time.time()-start
print 'GPU propagation and copy:', gputime

start=time.time()
v_post=v_pre
for _ in xrange(repeats):
    for i in spikes:
        v_post+=W[i, :]
cputime=time.time()-start
print 'CPU propagation:', cputime, '('+str(int(cputime/gputime))+'x)'

print 'Max abs difference CPU-GPU:', numpy.amax(numpy.abs(v_post_gpu-v_post))
