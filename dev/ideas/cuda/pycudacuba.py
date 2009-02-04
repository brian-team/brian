import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import bisect
import numpy, pylab, time
from itertools import repeat

N = 512*8
blocksize = 512
block = (blocksize,1,1)
grid = (N/blocksize,1)
duration = 1000
record = True

Ne = int(N*0.8)
Ni = N-Ne

mod = drv.SourceModule("""
__global__ void stateupdate(double *V, double *ge, double *gi)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double V__tmp = (ge[i]+gi[i]-(V[i]+0.049))/0.02;
    double ge__tmp = -ge[i]/0.005;
    double gi__tmp = -gi[i]/0.01;
    V[i] += 0.0001*V__tmp;
    ge[i] += 0.0001*ge__tmp;
    gi[i] += 0.0001*gi__tmp;
}

__global__ void threshold(double *V, int *spikes, bool *spiked, unsigned int *global_j, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool this_spiked = V[i]>-0.05; 
    spiked[i] = this_spiked;
    if(this_spiked) // && i<N) // can leave out i<N check if N%blocksize=0
    {
        unsigned int j = atomicInc(global_j, N);
        spikes[j] = i;
    }
}

__global__ void propagate(int *spikes, int numspikes, double *v, double *W, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int j=0; j<numspikes; j++)
        v[i] += W[i+N*spikes[j]];
}

__global__ void reset(double *V, bool *spiked)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    V[i] = (V[i]*!spiked[i])+(-0.06)*spiked[i]; // i.e. V[i]=-0.06 if spiked[i]
}
""")
stateupdate = mod.get_function("stateupdate")
threshold = mod.get_function("threshold")
propagate = mod.get_function("propagate")
reset = mod.get_function("reset")

V = gpuarray.to_gpu(numpy.random.rand(N)*0.01-0.06)
ge = gpuarray.to_gpu(numpy.zeros(N))
gi = gpuarray.to_gpu(numpy.zeros(N))

we = 0.00162
wi = -0.009
CeW = (numpy.random.rand(Ne, N)<0.02)*we
CiW = (numpy.random.rand(Ni, N)<0.02)*wi
CeW = gpuarray.to_gpu(CeW)
CiW = gpuarray.to_gpu(CiW)

gpu_spikes = drv.mem_alloc(4*N)
gpu_spikes_exc = drv.mem_alloc(4*N)
gpu_spikes_inh = drv.mem_alloc(4*N)
gpu_spiked = gpuarray.to_gpu(numpy.zeros(N, dtype=bool))
gpu_spike_index = drv.mem_alloc(4)
master_spikes = numpy.zeros(N, dtype=int)
spike_index = numpy.zeros(1, dtype=numpy.uint32)
drv.memcpy_htod(gpu_spikes, master_spikes)
drv.memcpy_htod(gpu_spike_index, spike_index)

if record:
    recspikes = []
    recv0 = numpy.zeros(duration)
totalnspikes = 0
start = time.time()
for t in xrange(duration):
    stateupdate(V, ge, gi, block=block, grid=grid)
    spike_index[0] = 0
    drv.memcpy_htod(gpu_spike_index, spike_index)
    threshold(V, gpu_spikes, gpu_spiked, gpu_spike_index, numpy.int32(N), block=block, grid=grid)
    drv.memcpy_dtoh(spike_index, gpu_spike_index)
    numspikes = spike_index[0]
    spikes = master_spikes[:numspikes]
    drv.memcpy_dtoh(spikes, gpu_spikes)
    numspikes_exc = bisect.bisect_left(spikes, Ne)
    numspikes_inh = numspikes - numspikes_exc
    drv.memcpy_htod(gpu_spikes_exc, spikes[:numspikes_exc])
    drv.memcpy_htod(gpu_spikes_inh, spikes[numspikes_exc:]-Ne)
    propagate(gpu_spikes_exc, numpy.int32(numspikes_exc), ge, CeW, numpy.int32(N), block=block, grid=grid)
    propagate(gpu_spikes_inh, numpy.int32(numspikes_inh), gi, CiW, numpy.int32(N), block=block, grid=grid)
    reset(V, gpu_spiked, block=block, grid=grid)
    if record:
        recspikes += zip(spikes, repeat(t))
        drv.memcpy_dtoh(recv0[t:t+1], V.gpudata)
        #recv0[t:t+1] = V[0]
    totalnspikes += numspikes
end = time.time()
print 'GPU time:', end-start
print 'Number of spikes:', totalnspikes

if record:
    i, t = zip(*recspikes)
    pylab.subplot(121)
    pylab.plot(t, i, '.')
    pylab.subplot(122)
    pylab.plot(recv0)
    pylab.show()