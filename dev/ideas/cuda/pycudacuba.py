import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import bisect
import numpy, pylab, time
from itertools import repeat
import cProfile as profile
import pstats

N=512*3200
blocksize=512
duration=10000
record=False
doprofile=False
if drv.get_version()==(2, 0, 0): # cuda version
    precision='float'
elif drv.get_version()>(2, 0, 0):
    precision='double'
else:
    raise Exception, "CUDA 2.0 required"

if precision=='double':
    mydtype=numpy.float64
else:
    mydtype=numpy.float32

block=(blocksize, 1, 1)
grid=(N/blocksize, 1)
Ne=int(N*0.8)
Ni=N-Ne

mod=drv.SourceModule("""
__global__ void stateupdate(SCALAR *V_arr, SCALAR *ge_arr, SCALAR *gi_arr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    SCALAR V = V_arr[i];
    SCALAR ge = ge_arr[i];
    SCALAR gi = gi_arr[i]; 
    SCALAR V__tmp = (ge+gi-(V+0.049))/0.02;
    SCALAR ge__tmp = -ge/0.005;
    SCALAR gi__tmp = -gi/0.01;
    V_arr[i] = V+0.0001*V__tmp;
    ge_arr[i] = ge+0.0001*ge__tmp;
    gi_arr[i] = gi+0.0001*gi__tmp;
}

__global__ void threshold(SCALAR *V, int *spikes, bool *spiked, unsigned int *global_j, int N)
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

__global__ void propagate(int *spikes, int numspikes, SCALAR *v, SCALAR *W, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    SCALAR v_tmp = v[i];
    for(int j=0; j<numspikes; j++)
        v_tmp += W[i+N*spikes[j]];
    v[i] = v_tmp;
}

__global__ void reset(SCALAR *V, bool *spiked)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool has_spiked = spiked[i];
    V[i] = (V[i]*!has_spiked)+(-0.06)*has_spiked; // i.e. V[i]=-0.06 if spiked[i]
}
""".replace('SCALAR', precision))
stateupdate=mod.get_function("stateupdate")
threshold=mod.get_function("threshold")
propagate=mod.get_function("propagate")
reset=mod.get_function("reset")

V=gpuarray.to_gpu(numpy.array(numpy.random.rand(N)*0.01-0.06, dtype=mydtype))
ge=gpuarray.to_gpu(numpy.zeros(N, dtype=mydtype))
gi=gpuarray.to_gpu(numpy.zeros(N, dtype=mydtype))

#we = 0.00162
#wi = -0.009
#pconn = 80./N
#CeW = numpy.array((numpy.random.rand(Ne, N)<pconn)*we,dtype=mydtype)
#CiW = numpy.array((numpy.random.rand(Ni, N)<pconn)*wi,dtype=mydtype)
#CeW = gpuarray.to_gpu(CeW)
#CiW = gpuarray.to_gpu(CiW)
#
#gpu_spikes = drv.mem_alloc(4*N)
#gpu_spikes_exc = drv.mem_alloc(4*N)
#gpu_spikes_inh = drv.mem_alloc(4*N)
#gpu_spiked = gpuarray.to_gpu(numpy.zeros(N, dtype=bool))
#gpu_spike_index = drv.mem_alloc(4)
#master_spikes = numpy.zeros(N, dtype=int)
#spike_index = numpy.zeros(1, dtype=numpy.uint32)
#drv.memcpy_htod(gpu_spikes, master_spikes)
#drv.memcpy_htod(gpu_spike_index, spike_index)
#
#if record:
#    recspikes = []
#    recv0 = numpy.zeros(duration,dtype=mydtype)
#totalnspikes = 0

stateupdate.prepare(('i', 'i', 'i'), block)
#threshold.prepare(('i', 'i', 'i', 'i', numpy.int32), block)
#propagate.prepare(('i', numpy.int32, 'i', 'i', numpy.int32), block)
#reset.prepare(('i', 'i'), block)

stateupdate_args=(grid, int(V.gpudata), int(ge.gpudata), int(gi.gpudata))
#threshold_args = (grid, int(V.gpudata), int(gpu_spikes), int(gpu_spiked.gpudata), int(gpu_spike_index), numpy.int32(N))
#reset_args = (grid, int(V.gpudata), int(gpu_spiked.gpudata))

def run_sim():
    global totalnspikes
    for t in xrange(duration):
        #stateupdate(V, ge, gi, block=block, grid=grid)
        if t==0:
            stateupdate.prepared_call(*stateupdate_args)
        else:
            stateupdate.launch_grid(*grid)
#        spike_index[0] = 0
#        drv.memcpy_htod(gpu_spike_index, spike_index)
#        #threshold(V, gpu_spikes, gpu_spiked, gpu_spike_index, numpy.int32(N), block=block, grid=grid)
#        if t==0:
#            threshold.prepared_call(*threshold_args)
#        else:
#            threshold.launch_grid(*grid)
#        drv.memcpy_dtoh(spike_index, gpu_spike_index)
#        numspikes = spike_index[0]
#        spikes = master_spikes[:numspikes]
#        drv.memcpy_dtoh(spikes, gpu_spikes)
#        spikes.sort()
#        numspikes_exc = bisect.bisect_left(spikes, Ne)
#        numspikes_inh = numspikes - numspikes_exc
#        drv.memcpy_htod(gpu_spikes_exc, spikes[:numspikes_exc])
#        drv.memcpy_htod(gpu_spikes_inh, spikes[numspikes_exc:]-Ne)
#        ###propagate(gpu_spikes_exc, numpy.int32(numspikes_exc), ge, CeW, numpy.int32(N), block=block, grid=grid)
#        propagate.prepared_call(grid, int(gpu_spikes_exc), numpy.int32(numspikes_exc), int(ge.gpudata), int(CeW.gpudata), numpy.int32(N))
#        ###propagate(gpu_spikes_inh, numpy.int32(numspikes_inh), gi, CiW, numpy.int32(N), block=block, grid=grid)
#        propagate.prepared_call(grid, int(gpu_spikes_inh), numpy.int32(numspikes_inh), int(gi.gpudata), int(CiW.gpudata), numpy.int32(N))
#        ###reset(V, gpu_spiked, block=block, grid=grid)
#        if t==0:
#            reset.prepared_call(*reset_args)
#        else:
#            reset.launch_grid(*grid)
#        if record:
#            recspikes += zip(spikes, repeat(t))
#            drv.memcpy_dtoh(recv0[t:t+1], V.gpudata)
#        totalnspikes += numspikes

if doprofile:
    start=time.time()
    profile.run('run_sim()', 'cudacuba.prof')
    end=time.time()
    stats=pstats.Stats('cudacuba.prof')
    #stats.strip_dirs()
    stats.sort_stats('cumulative', 'calls')
    stats.print_stats(50)
else:
    start=time.time()
    run_sim()
    end=time.time()
print 'N:', N
print 'GPU time:', end-start
print 'Number of spikes:', totalnspikes

if record:
    i, t=zip(*recspikes)
    pylab.subplot(121)
    pylab.plot(t, i, '.')
    pylab.subplot(122)
    pylab.plot(recv0)
    pylab.show()
