import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import bisect
import numpy, pylab, time
from itertools import repeat
import cProfile as profile
import pstats
import brian

N = 512 * 256
blocksize = 512
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

mod = drv.SourceModule("""
__global__ void stateupdate(SCALAR *V, SCALAR *ge, SCALAR *gi)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    SCALAR V__tmp = (ge[i]+gi[i]-(V[i]+0.049))/0.02;
    SCALAR ge__tmp = -ge[i]/0.005;
    SCALAR gi__tmp = -gi[i]/0.01;
    V[i] += 0.0001*V__tmp;
    ge[i] += 0.0001*ge__tmp;
    gi[i] += 0.0001*gi__tmp;
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

__global__ void single_thread_threshold(SCALAR *V, int *spikes, bool *spiked, unsigned int *global_j, int N)
{
    unsigned int j = 0;
    for(int i=0;i<N;i++)
    {
        if(V[i]>-0.05)
        {
            spiked[i] = true;
            spikes[j++] = i;
        } else
        {
            spiked[i] = false;
        }
    }
    *global_j = j;
}

__global__ void propagate(int *spikes, int numspikes, SCALAR *v, SCALAR *W, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int j=0; j<numspikes; j++)
        v[i] += W[i+N*spikes[j]];
}

__global__ void propagate_spike(int spike, SCALAR *alldata, int *allj, int *rowind, SCALAR *V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int startindex = rowind[spike];
    if(i>=rowind[spike+1]-startindex) return;
    i += startindex;
    V[allj[i]] += alldata[i];
}

__global__ void reset(SCALAR *V, bool *spiked)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    V[i] = (V[i]*!spiked[i])+(-0.06)*spiked[i]; // i.e. V[i]=-0.06 if spiked[i]
}
""".replace('SCALAR', precision))
stateupdate = mod.get_function("stateupdate")
threshold = mod.get_function("threshold")
single_thread_threshold = mod.get_function("single_thread_threshold")
propagate = mod.get_function("propagate")
propagate_spike = mod.get_function("propagate_spike")
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

gpu_spikes = drv.mem_alloc(4 * N)
gpu_spikes_exc = drv.mem_alloc(4 * N)
gpu_spikes_inh = drv.mem_alloc(4 * N)
gpu_spiked = gpuarray.to_gpu(numpy.zeros(N, dtype=bool))
gpu_spike_index = drv.mem_alloc(4)
master_spikes = numpy.zeros(N, dtype=int)
spike_index = numpy.zeros(1, dtype=numpy.uint32)
drv.memcpy_htod(gpu_spikes, master_spikes)
drv.memcpy_htod(gpu_spike_index, spike_index)

recspikes = []
recv0 = numpy.zeros(duration, dtype=mydtype)
totalnspikes = 0

stateupdate.prepare(('i', 'i', 'i'), block)
threshold.prepare(('i', 'i', 'i', 'i', numpy.int32), block)
single_thread_threshold.prepare(('i', 'i', 'i', 'i', numpy.int32), (1, 1, 1))
propagate.prepare(('i', numpy.int32, 'i', 'i', numpy.int32), block)
propagate_spike.prepare((numpy.int32, 'i', 'i', 'i', 'i'), block)
reset.prepare(('i', 'i'), block)

stateupdate_args = (grid, int(V.gpudata), int(ge.gpudata), int(gi.gpudata))
threshold_args = (grid, int(V.gpudata), int(gpu_spikes), int(gpu_spiked.gpudata), int(gpu_spike_index), numpy.int32(N))
single_thread_threshold_args = ((1, 1), int(V.gpudata), int(gpu_spikes), int(gpu_spiked.gpudata), int(gpu_spike_index), numpy.int32(N))
reset_args = (grid, int(V.gpudata), int(gpu_spiked.gpudata))
propagate_spike_end_args_exc = (int(Ce_alldata), int(Ce_allj), int(Ce_rowind), int(ge.gpudata))
propagate_spike_end_args_inh = (int(Ci_alldata), int(Ci_allj), int(Ci_rowind), int(gi.gpudata))

def run_sim():
    global totalnspikes, recspikes, recv0
    for t in xrange(duration):
        #stateupdate(V, ge, gi, block=block, grid=grid)
        if t == 0:
            stateupdate.prepared_call(*stateupdate_args)
        else:
            stateupdate.launch_grid(*grid)
        spike_index[0] = 0
        drv.memcpy_htod(gpu_spike_index, spike_index)
        #threshold(V, gpu_spikes, gpu_spiked, gpu_spike_index, numpy.int32(N), block=block, grid=grid)
        if t == 0:
            threshold.prepared_call(*threshold_args)
        else:
            threshold.launch_grid(*grid)
        # MUCH! slower
        #single_thread_threshold.prepared_call(*single_thread_threshold_args)
        drv.memcpy_dtoh(spike_index, gpu_spike_index)
        numspikes = spike_index[0]
        spikes = master_spikes[:numspikes]
        drv.memcpy_dtoh(spikes, gpu_spikes)
        spikes.sort()
        numspikes_exc = bisect.bisect_left(spikes, Ne)
        numspikes_inh = numspikes - numspikes_exc
        #drv.memcpy_htod(gpu_spikes_exc, spikes[:numspikes_exc])
        #drv.memcpy_htod(gpu_spikes_inh, spikes[numspikes_exc:]-Ne)
        #propagate(gpu_spikes_exc, numpy.int32(numspikes_exc), ge, CeW, numpy.int32(N), block=block, grid=grid)
        #propagate.prepared_call(grid, int(gpu_spikes_exc), numpy.int32(numspikes_exc), int(ge.gpudata), int(CeW.gpudata), numpy.int32(N))
        #propagate(gpu_spikes_inh, numpy.int32(numspikes_inh), gi, CiW, numpy.int32(N), block=block, grid=grid)
        #propagate.prepared_call(grid, int(gpu_spikes_inh), numpy.int32(numspikes_inh), int(gi.gpudata), int(CiW.gpudata), numpy.int32(N))
        for i in spikes[:numspikes_exc]:
            ncols = len(Ce.rowj[i])
            if ncols % 512 == 0:
                spikegrid = (ncols / 512, 1)
            else:
                spikegrid = (ncols / 512 + 1, 1)
            propagate_spike.prepared_call(spikegrid, numpy.int32(i), *propagate_spike_end_args_exc)
        for i in spikes[numspikes_exc:] - Ne:
            ncols = len(Ci.rowj[i])
            if ncols % 512 == 0:
                spikegrid = (ncols / 512, 1)
            else:
                spikegrid = (ncols / 512 + 1, 1)
            propagate_spike.prepared_call(spikegrid, numpy.int32(i), *propagate_spike_end_args_inh)
        #reset(V, gpu_spiked, block=block, grid=grid)
        if t == 0:
            reset.prepared_call(*reset_args)
        else:
            reset.launch_grid(*grid)
        if record:
            recspikes += zip(spikes, repeat(t))
            drv.memcpy_dtoh(recv0[t:t + 1], V.gpudata)
        totalnspikes += numspikes

if doprofile:
    start = time.time()
    profile.run('run_sim()', 'cudacuba.prof')
    end = time.time()
    stats = pstats.Stats('cudacuba.prof')
    #stats.strip_dirs()
    stats.sort_stats('cumulative', 'calls')
    stats.print_stats(50)
else:
    start = time.time()
    run_sim()
    end = time.time()
print 'GPU time:', end - start
print 'Number of spikes:', totalnspikes

if record:
    i, t = zip(*recspikes)
    pylab.subplot(121)
    pylab.plot(t, i, '.')
    pylab.subplot(122)
    pylab.plot(recv0)
    pylab.show()
