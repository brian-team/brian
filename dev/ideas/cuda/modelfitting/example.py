import pycuda.autoinit as autoinit
import pycuda.driver as drv
import pycuda
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import bisect
import numpy, pylab, time
from matplotlib.cm import jet
major, minor = pycuda.VERSION
if minor>92:
    from pycuda.compiler import SourceModule
else:
    from pycuda.driver import SourceModule

N = 10000
blocksize = 256 # too many registers for 512
doplot = True

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

mod = SourceModule("""
__global__ void runsim(
                    SCALAR *V_arr,            // State variables for each neuron
                    SCALAR *tau_arr,
                    SCALAR *R_arr,
                    SCALAR *I_arr,            // Input current
                    int *num_coincidences,    // Count of coincidences for each neuron
                    int *spiketimes,          // Array of all spike times as integers
                    int *spiketime_indices,   // Pointer into above array for each neuron
                    int Tstart, int Tend      // Start, end time as integer (t=T*dt)
                    )
{
    const int neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    // Load variables at start
    SCALAR V = V_arr[neuron_index];
    SCALAR tau = tau_arr[neuron_index];
    SCALAR R = R_arr[neuron_index];
    int spiketime_index = spiketime_indices[neuron_index];
    int last_spike_time = spiketimes[spiketime_index];
    int next_spike_time = spiketimes[spiketime_index+1];
    int ncoinc = num_coincidences[neuron_index];
    for(int T=Tstart; T<Tend; T++)
    {
        SCALAR t = T*_dt_;
        // Read input current
        SCALAR I = I_arr[T]; // this is a global read for each thread, can maybe
                             // reduce this by having just one read per block,
                             // put it into shared memory, and then have all the
                             // threads in that block read it, we could even
                             // maybe buffer reads of I into shared memory -
                             // experiment with this 
        // State update
        SCALAR V__tmp = (-V+R*I)/tau;
        V += V+_dt_*V__tmp;
        // Threshold
        const bool has_spiked = V>1.;     // Vt=1
        // Reset
        V = V*!has_spiked;                // Vr=0
        // Coincidence counter
        ncoinc += has_spiked && (((last_spike_time+_delta_)>=T) || ((next_spike_time-_delta_)<=T));
        if(T==next_spike_time){           // divergence for each input spike
            spiketime_index++;
            last_spike_time = next_spike_time;
            next_spike_time = spiketimes[spiketime_index+1];
        }
    }
    // Store variables at end
    V_arr[neuron_index] = V;
    spiketime_indices[neuron_index] = spiketime_index;
    num_coincidences[neuron_index] = ncoinc;
}
""".replace('SCALAR', precision).replace('_dt_', '0.0001').replace('_delta_', '40'))
runsim = mod.get_function("runsim")

# Read data from files
I = numpy.loadtxt('current.txt')
duration = len(I)
spiketimes = numpy.array((numpy.loadtxt('spikes.txt')-32.)/0.0001, dtype=int)
# we add a -10000 at the beginning as there is no previous spike at the beginning,
# and one at the end because the kernel moves to the next spike by checking for the
# existence of the current spike 
spiketimes = numpy.hstack((-10000, spiketimes, -10000))

V = gpuarray.to_gpu(numpy.zeros(N, dtype=mydtype))
R = gpuarray.to_gpu(numpy.array(numpy.random.rand(N)*1e9, dtype=mydtype))
tau = gpuarray.to_gpu(numpy.array(numpy.random.rand(N)*0.100+0.001, dtype=mydtype))
I = gpuarray.to_gpu(numpy.array(I, dtype=mydtype))
num_coincidences = gpuarray.to_gpu(numpy.zeros(N, dtype=int))
spiketime_indices = gpuarray.to_gpu(numpy.zeros(N, dtype=int))
spiketimes = gpuarray.to_gpu(numpy.array(spiketimes, dtype=int))

#runsim.prepare(('i', 'i', 'i', 'i'), block)
#stateupdate_args = (grid, int(V.gpudata), int(ge.gpudata), int(gi.gpudata))
#stateupdate.prepared_call(*stateupdate_args)

start_time = time.time()
runsim(V, R, tau,
       I,
       num_coincidences,
       spiketimes,
       spiketime_indices,
       numpy.int32(0), numpy.int32(duration),
       block=block, grid=grid
       )
autoinit.context.synchronize()
running_time = time.time()-start_time

print 'N:', N, 'blocksize:', blocksize
print 'Duration:', duration, 'steps =', duration*0.0001, 's'
print 'Total running time:', running_time

if doplot:
    num_coincidences = num_coincidences.get()
    R = R.get()
    tau = tau.get()
    
    max_coinc = float(num_coincidences.max())
    
    pylab.scatter(R, tau, color=jet(num_coincidences/max_coinc))
    pylab.xlabel('R')
    pylab.ylabel('tau')
    pylab.title('Hot = '+str(max_coinc)+' coincidences, cool = 0')
    pylab.show()