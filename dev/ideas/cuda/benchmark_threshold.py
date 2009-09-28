import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import bisect
import numpy, pylab, time, random
from scipy import weave

N = 1024
nspike = int(0.1*N)
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
Ngrid = N/blocksize
grid = (Ngrid,1)
grid1 = (1, 1)

mod = drv.SourceModule("""
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
    if(this_spiked)
    {
        spikes[atomicInc(global_j, N)] = i;
    }
}

/////// TODO: next three functions are untested, but should be a non-blocking threshold function if N<=512*512

__global__ void threshold_blocksumcount(SCALAR *V, unsigned int *blocksumcount)
{ 
    __shared__ unsigned int partialsum[BLOCKSIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    int b = blockIdx.x;
    bool this_spiked = V[i]>-0.05;
    partialsum[t] = (unsigned int)this_spiked;
    for(unsigned int stride=blockDim.x/2; stride>=1; stride/=2)
    {
        __syncthreads();
        if(t<stride)
            partialsum[t] += partialsum[t+stride];
    }
    __syncthreads();
    if(t==0)
        blocksumcount[b] = partialsum[t];
}

__global__ void threshold_cumsum(unsigned int *blocksumcount, unsigned int *cumblocksumcount)
{
    __shared__ unsigned int partialsum[BLOCKSIZE];
    int t = threadIdx.x;
    partialsum[t] = blocksumcount[t];
    for(unsigned int stride=1; stride<blockDim.x; stride*=2)
    {
        __syncthreads();
        if(t>=stride)
            partialsum[t] += partialsum[t-stride];
    }
    cumblocksumcount[t+1] = partialsum[t];
}

__global__ void threshold_compact(SCALAR *V, int *spikes, unsigned int *blocksumcount, unsigned int *cumblocksumcount)
{
    __shared__ unsigned int partialsum[BLOCKSIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    int b = blockIdx.x;
    // only compact those blocks with some spikes in (many will not in general, and this introduces no divergence)
    if(blocksumcount[b]>0)
    {
        bool this_spiked = V[i]>-0.05;
        partialsum[t] = (unsigned int)this_spiked;
        for(unsigned int stride=1; stride<blockDim.x; stride*=2)
        {
            __syncthreads();
            if(t>=stride)
                partialsum[t] += partialsum[t-stride];
        }
        __syncthreads();
        if(this_spiked)
          spikes[partialsum[t]+cumblocksumcount[b]-1] = i;
        //spikes[i] = partialsum[t]+cumblocksumcount[b];
    }
}
""".replace('SCALAR',precision).replace('BLOCKSIZE', str(blocksize)))
stateupdate = mod.get_function("stateupdate")
threshold = mod.get_function("threshold")
threshold_blocksumcount = mod.get_function("threshold_blocksumcount")
threshold_cumsum = mod.get_function("threshold_cumsum")
threshold_compact = mod.get_function("threshold_compact")

V_cpu = numpy.zeros(N, dtype=mydtype)
V_cpu[:nspike] = -0.04 # above threshold
V_cpu[nspike:] = -0.06 # below threshold
random.shuffle(V_cpu)
V = gpuarray.to_gpu(V_cpu)
ge = gpuarray.to_gpu(numpy.zeros(N,dtype=mydtype))
gi = gpuarray.to_gpu(numpy.zeros(N,dtype=mydtype))

gpu_spikes = drv.mem_alloc(4*N)
gpu_spiked = gpuarray.to_gpu(numpy.zeros(N, dtype=bool))
gpu_spike_index = drv.mem_alloc(4)
spikes = master_spikes = numpy.zeros(N, dtype=int)
spike_index = numpy.zeros(1, dtype=numpy.uint32)
gpu_blocksumcount = gpuarray.to_gpu(numpy.zeros(Ngrid, dtype=numpy.uint32))
gpu_cumblocksumcount = gpuarray.to_gpu(numpy.zeros(Ngrid+1, dtype=numpy.uint32))
drv.memcpy_htod(gpu_spikes, master_spikes)
drv.memcpy_htod(gpu_spike_index, spike_index)

stateupdate.prepare(('i', 'i', 'i'), block)
threshold.prepare(('i', 'i', 'i', 'i', numpy.int32), block)
threshold_blocksumcount.prepare(('i', 'i'), block)
threshold_cumsum.prepare(('i', 'i'), block)
threshold_compact.prepare(('i', 'i', 'i', 'i'), block)

stateupdate_args = (grid, int(V.gpudata), int(ge.gpudata), int(gi.gpudata))
threshold_args = (grid, int(V.gpudata), int(gpu_spikes), int(gpu_spiked.gpudata), int(gpu_spike_index), numpy.int32(N))
threshold_blocksumcount_args = (grid, int(V.gpudata), int(gpu_blocksumcount.gpudata))
threshold_cumsum_args = (grid1, int(gpu_blocksumcount.gpudata), int(gpu_cumblocksumcount.gpudata))
threshold_compact_args = (grid, int(V.gpudata), int(gpu_spikes), int(gpu_blocksumcount.gpudata), int(gpu_cumblocksumcount.gpudata))
stateupdate.prepared_call(*stateupdate_args)
threshold.prepared_call(*threshold_args)

print 'before blocksumcount'
threshold_blocksumcount.prepared_call(*threshold_blocksumcount_args)
autoinit.context.synchronize()
print gpu_blocksumcount.get()
print sum(V_cpu[:512]>-0.05), sum(V_cpu[512:1024]>-0.05) # these two should be the same...

print numpy.cumsum(gpu_blocksumcount.get())
print 'before cumsum'
threshold_cumsum.prepared_call(*threshold_cumsum_args)
autoinit.context.synchronize()
print gpu_cumblocksumcount.get()
print 'before compact'
threshold_compact.prepared_call(*threshold_compact_args)
autoinit.context.synchronize()

drv.memcpy_dtoh(spikes, gpu_spikes)
print spikes[:102]
print gpu_cumblocksumcount.get()[-1]
print sum(V_cpu>-0.05)
S1 = spikes[:gpu_cumblocksumcount.get()[-1]]
S2 = (V_cpu>-0.05).nonzero()[0]
S1.sort()
S2.sort()
print S1
print S2
print (S1==S2).all() # TODO: not quite there yet...

exit()

def run_sim():
    for t in xrange(duration):
        spike_index[0] = 0
        drv.memcpy_htod(gpu_spike_index, spike_index)
        threshold.launch_grid(*grid)

def run_sim_cpu():
    for t in xrange(duration):
        Vt = -0.05
        code =  """
                int numspikes=0;
                for(int i=0;i<N;i++)
                    if(V_cpu(i)>Vt)
                        spikes(numspikes++) = i;
                return_val = numspikes;
                """
        numspikes = weave.inline(code,['spikes','V_cpu','Vt','N'],
                                 compiler='gcc',
                                 type_converters=weave.converters.blitz)

start = time.time()
run_sim()
timetaken_gpu = time.time()-start

start = time.time()
run_sim_cpu()
timetaken_cpu = time.time()-start

print 'N:', N, 'nspike', nspike, 'blocksize:', blocksize, 'numsteps:', duration
print 'GPU time:', timetaken_gpu
print 'CPU time:', timetaken_cpu
