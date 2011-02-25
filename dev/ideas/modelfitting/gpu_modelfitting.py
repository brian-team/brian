from brian import *
from playdoh import *
import brian.optimiser as optimiser
#import pycuda.autoinit as autoinit
import pycuda.driver as drv
import pycuda
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
try:
    from pycuda.compiler import SourceModule
except ImportError:
    from pycuda.driver import SourceModule
from numpy import *
from brian.experimental.codegen.integration_schemes import *
from brian.experimental.codegen.codegen_gpu import *
import re

__all__ = ['GPUModelFitting',
           'euler_scheme', 'rk2_scheme', 'exp_euler_scheme'
           ]

if drv.get_version() == (2, 0, 0): # cuda version
    default_precision = 'float'
elif drv.get_version() > (2, 0, 0):
    default_precision = 'double'
else:
    raise Exception, "CUDA 2.0 required"

class ModelfittingGPUCodeGenerator(GPUCodeGenerator):
    def generate(self, eqs, scheme):
        vartype = self.vartype()
        code = ''
        for line in self.scheme(eqs, scheme).split('\n'):
            line = line.strip()
            if line:
                code += '    ' + line + '\n'
        return code




def get_cuda_template():
    return """
    __global__ void runsim(
        // ITERATIONS
        int Tstart, int Tend,             // Start, end time as integer (t=T*dt)
        
        // STATE VARIABLES
        %SCALAR% *state_vars,             // State variables are offset from this
        
        // INPUT PARAMETERS
        double *I_arr,                    // Input current
        int *I_arr_offset,                // Input current offset (for separate input
                                          // currents for each neuron)
        // DELAYS PARAMETERS
        int *spikedelay_arr,              // Integer delay for each spike
        
        // REFRACTORY PARAMETERS
        int *refractory_arr,              // Integer refractory times
        int *next_allowed_spiketime_arr,  // Integer time of the next allowed spike (for refractoriness)
        
        // CRITERION SPECIFIC PARAMETERS
        %CRITERION_DECLARE%
        
        // SPIKES PARAMETERS
        %SPIKES_DECLARE%
        
        // TRACES PARAMETERS
        %TRACES_DECLARE%
        
        // MISC PARAMETERS
        int onset                         // Time onset (only count spikes from here onwards)
        )
    {
        // NEURON INDEX
        const int neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
        if(neuron_index>=%NUM_NEURONS%) return;
        
        // EXTRACT STATE VARIABLES
        %EXTRACT_STATE_VARIABLES%
        
        // LOAD VARIABLES
        %LOAD_VARIABLES%
        
        // SPIKE INITIALIZATION
        %SPIKES_INIT%
        //int spiketime_index = spiketime_indices[neuron_index];
        
        // CRITERION INITIALIZATION
        %CRITERION_INIT%
        //int last_spike_time = spiketimes[spiketime_index];
        //int next_spike_time = spiketimes[spiketime_index+1];
        //int ncoinc = num_coincidences[neuron_index];
        //int nspikes = spikecount[neuron_index];
        //%COINCIDENCE_COUNT_INIT%
        
        // INPUT INITIALIZATION
        int I_offset = I_arr_offset[neuron_index];
        
        // DELAYS INITIALIZATION
        int spikedelay = spikedelay_arr[neuron_index];
        
        // REFRACTORY INITIALIZATION
        const int refractory = refractory_arr[neuron_index];
        int next_allowed_spiketime = next_allowed_spiketime_arr[neuron_index];
        
        for(int T=Tstart; T<Tend; T++)
        {
            %SCALAR% t = T*%DT%;
            // Read input current
            %SCALAR% ${input_var} = I_arr[T+I_offset];
                                 // this is a global read for each thread, can maybe
                                 // reduce this by having just one read per block,
                                 // put it into shared memory, and then have all the
                                 // threads in that block read it, we could even
                                 // maybe buffer reads of I into shared memory -
                                 // experiment with this
                                 
            // STATE UPDATE
            %STATE_UPDATE%
            
            // THRESHOLD
            const bool is_refractory = (T<=next_allowed_spiketime);
            const bool has_spiked = (%THRESHOLD%)&&!is_refractory;
            
            // RESET
            if(has_spiked||is_refractory)
            {
                %RESET%
            }
            
            if(has_spiked)
                next_allowed_spiketime = T+refractory;
            
            // CRITERION TIMESTEP
            %CRITERION_TIMESTEP%
            // Coincidence counter
            //const int Tspike = T+spikedelay;
            //%COINCIDENCE_COUNT_TEST%
            //nspikes += has_spiked*(T>=onset);
            //if(Tspike>=next_spike_time){
            //    spiketime_index++;
            //    last_spike_time = next_spike_time;
            //    next_spike_time = spiketimes[spiketime_index+1];
            //    %COINCIDENCE_COUNT_NEXT%
            //}
        }
        // Store variables at end
        %STORE_VARIABLES%
        
        // CRITERION END
        %CRITERION_END%
        //%COINCIDENCE_COUNT_STORE_VARIABLES%
        //num_coincidences[neuron_index] = ncoinc;
        //spikecount[neuron_index] = nspikes;
        
        next_allowed_spiketime_arr[neuron_index] = next_allowed_spiketime;
        
        // SPIKE END
        %SPIKES_END%
        //spiketime_indices[neuron_index] = spiketime_index;
    }
    """

def generate_modelfitting_kernel_src(G, eqs, threshold, reset, dt, num_neurons,
                                     criterion,
                                     precision=default_precision,
                                     scheme=euler_scheme
                                     ):
    eqs.prepare()
    src = modelfitting_kernel_template
    # Substitute state variable declarations
    indexvar = dict((v, k) for k, v in G.var_index.iteritems() if isinstance(k, str) and k!='I')
    extractions = '\n    '.join('%SCALAR% *'+name+'_arr = state_vars+'+str(i*num_neurons)+';' for i, name in indexvar.iteritems())
    src = src.replace('%EXTRACT_STATE_VARIABLES%', extractions)
    # Substitute load variables
    loadvar_names = eqs._diffeq_names + []
    loadvar_names.remove('I') # I is assumed to be a parameter and loaded per time step
    loadvars = '\n    '.join('%SCALAR% ' + name + ' = ' + name + '_arr[neuron_index];' for name in loadvar_names)
    src = src.replace('%LOAD_VARIABLES%', loadvars)
    # Substitute save variables
    savevars = '\n    '.join(name + '_arr[neuron_index] = ' + name + ';' for name in loadvar_names)
    src = src.replace('%STORE_VARIABLES%', savevars)
    # Substitute threshold
    src = src.replace('%THRESHOLD%', threshold)
    # Substitute reset
    reset = '\n            '.join(line.strip() + ';' for line in reset.split('\n') if line.strip())
    src = src.replace('%RESET%', reset)
    # Substitute state update
    sulines = ModelfittingGPUCodeGenerator(dtype=precision).generate(eqs, scheme)
    sulines = re.sub(r'\bdt\b', '%DT%', sulines)
    src = src.replace('%STATE_UPDATE%', sulines.strip())
    
    # TODO
    # Substitute coincidence counting algorithm
#    ccalgo = coincidence_counting_algorithm_src[coincidence_count_algorithm]
#    for search, replace in ccalgo.iteritems():
#        src = src.replace(search, replace)
    if criterion.type == 'spikes':
        spikes_declare = """
    int *spiketimes,          // Array of all spike times as integers (begin and
                              // end each train with large negative value)
    int *spiketime_indices,   // Pointer into above array for each neuron
        """
        src = src.replace('%SPIKES_DECLARE%', spikes_declare)
    if criterion.type == 'traces':
        # TODO: traces_declare = ...
        src = src.replace('%TRACES_DECLARE%', traces_declare)
        
    # Substitute dt
    src = src.replace('%DT%', str(float(dt)))
    # Substitute SCALAR
    src = src.replace('%SCALAR%', precision)
    # Substitute number of neurons
    src = src.replace('%NUM_NEURONS%', str(num_neurons))
    # Substitute delta, the coincidence window half-width
    src = src.replace('%DELTA%', str(int(rint(delta / dt))))
    return src




class GPUModelFitting(object):
    '''
    Model fitting class to interface with GPU
    
    Initialisation arguments:
    
    ``G``
        The initialised NeuronGroup to work from.
    ``eqs``
        The equations defining the NeuronGroup.
    ``I``, ``I_offset``
        The current array and offsets (see below).
    ``spiketimes``, ``spiketimes_offset``
        The spike times array and offsets (see below).
    ``spikedelays``,
        Array of delays for each neuron.
    ``refractory``,
        Array of refractory periods, or a single value.
    ``delta``
        The half-width of the coincidence window.
    ``precision``
        Should be 'float' or 'double' - by default the highest precision your
        GPU supports.
    ``coincidence_count_algorithm``
        Should be 'inclusive' if multiple predicted spikes can match one
        target spike, or 'exclusive' (default) if multiple predicted spikes
        can match only one target spike (earliest spikes are matched first). 
    
    Methods:
    
    ``reinit_vars(I, I_offset, spiketimes, spiketimes_offset, spikedelays)``
        Reinitialises all the variables, counters, etc. The state variable values
        are copied from the NeuronGroup G again, and the variables I, I_offset, etc.
        are copied from the method arguments.
    ``launch(duration[, stepsize])``
        Runs the kernel on the GPU for simulation time duration. If ``stepsize``
        is given, the simulation is broken into pieces of that size. This is
        useful on Windows because driver limitations mean that individual GPU kernel
        launches cannot last more than a few seconds without causing a crash.
    
    Attributes:
    
    ``coincidence_count``
        An array of the number of coincidences counted for each neuron.
    ``spike_count``
        An array of the number of spikes counted for each neuron.
    
    **Details**
    
    The equations for the NeuronGroup can be anything, but they will be solved
    with the Euler method. One restriction is that there must be a parameter
    named I which is the time varying input current. (TODO: support for
    multiple input currents? multiple names?)
    
    The current I is passed to the GPU in the following way. Define a single 1D
    array I which contains all the time varying current arrays one after the
    other. Now define an array I_offset of integers so that neuron i should see
    currents: I[I_offset[i]], I[I_offset[i]+1], I[I_offset[i]+2], etc.
    
    The experimentally recorded spike times should be passed in a similar way,
    put all the spike times in a single array and pass an offsets array
    spiketimes_offset. One difference is that it is essential that each spike
    train with the spiketimes array should begin with a spike at a
    large negative time (e.g. -1*second) and end with a spike that is a long time
    after the duration of the run (e.g. duration+1*second). The GPU uses this to
    mark the beginning and end of the train rather than storing the number of
    spikes for each train. 
    '''
    def __init__(self, G, eqs,
                 delta, onset=0*ms,
                 criterion, # Criterion object
                 precision=default_precision,
                 scheme=euler_scheme
                 ):
        eqs.prepare()
        self.precision = precision
        if precision == 'double':
            self.mydtype = float64
        else:
            self.mydtype = float32
        self.N = N = len(G)
        self.dt = dt = G.clock.dt
        self.delta = delta
        self.onset = onset
        self.eqs = eqs
        self.G = G
        self.criterion = criterion
        self.generate_code()

    def generate_threshold_code(self):
        eqs = self.eqs
        threshold = self.G._threshold
        if threshold.__class__ is Threshold:
            state = threshold.state
            if isinstance(state, int):
                state = eqs._diffeq_names[state]
            threshold = state + '>' + str(float(threshold.threshold))
        elif isinstance(threshold, VariableThreshold):
            state = threshold.state
            if isinstance(state, int):
                state = eqs._diffeq_names[state]
            threshold = state + '>' + threshold.threshold_state
        elif isinstance(threshold, StringThreshold):
            namespace = threshold._namespace
            expr = threshold._expr
            all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
            expr = optimiser.freeze(expr, all_variables, namespace)
            threshold = expr
        else:
            raise ValueError('Threshold must be constant, VariableThreshold or StringThreshold.')
        return threshold
        
    def generate_reset_code(self):
        eqs = self.eqs
        reset = self.G._resetfun
        if reset.__class__ is Reset:
            state = reset.state
            if isinstance(state, int):
                state = eqs._diffeq_names[state]
            reset = state + ' = ' + str(float(reset.resetvalue))
        elif isinstance(reset, VariableReset):
            state = reset.state
            if isinstance(state, int):
                state = eqs._diffeq_names[state]
            reset = state + ' = ' + reset.resetvaluestate
        elif isinstance(reset, StringReset):
            namespace = reset._namespace
            expr = reset._expr
            all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
            expr = optimiser.freeze(expr, all_variables, namespace)
            reset = expr
        return reset
            
    def generate_code(self):
        threshold = self.generate_threshold_code()
        reset = self.generate_reset_code()
        self.kernel_src = generate_modelfitting_kernel_src(
                  self.G, self.eqs, threshold, reset, self.dt, self.N, self.delta,
                  coincidence_count_algorithm=self.coincidence_count_algorithm,
                  precision=self.precision, scheme=self.scheme)

    def initialize_spikes(self, spiketimes, spiketimes_indices):
        self.spiketimes = gpuarray.to_gpu(array(rint(spiketimes / self.dt), dtype=int32))
        self.spiketime_indices = gpuarray.to_gpu(array(spiketimes_offset, dtype=int32))
        
    def initialize_traces(self):
        # TODO
        pass

    def initialize_delays(self, spikedelays):
        self.spikedelay_arr = gpuarray.to_gpu(array(rint(spikedelays / self.dt), dtype=int32))
    
    def initialize_refractory(self, refractory):
        if isinstance(refractory, float):
            refractory = refractory*ones(self.N)
        self.refractory_arr = gpuarray.to_gpu(array(rint(refractory / self.dt), dtype=int32))
        self.next_allowed_spiketime_arr = gpuarray.to_gpu(-ones(self.N, dtype=int32))

    def initialize_kernel_arguments(self):
        self.kernel_func_args = [self.statevars_arr,
                                 self.I,
                                 self.I_offset,
                                 self.spikedelay_arr,
                                 self.refractory_arr,
                                 self.next_allowed_spiketime_arr]
        self.kernel_func_args += self.criterion.get_kernel_arguments()
        if self.criterion.type == 'spikes':
            self.kernel_func_args += [self.spiketimes,
                                      self.spiketime_indices]
        if self.criterion.type == 'traces':
            # TODO
            self.kernel_func_args += []
        self.kernel_func_args += [int32(rint(self.onset / self.dt))]
        
        if self.coincidence_count_algorithm == 'exclusive':
            self.kernel_func_args += [self.last_spike_allowed_arr,
                                      self.next_spike_allowed_arr]

    def reinit_vars(self, I, I_offset,
                    spikedelays, refractory,
                    spiketimes = None, spiketimes_offset = None,
                    inputs = None, inputs_offset = None):
        self.kernel_module = SourceModule(self.kernel_src)
        self.kernel_func = self.kernel_module.get_function('runsim')
        blocksize = 128
        try:
            blocksize = self.kernel_func.get_attribute(pycuda.driver.function_attribute.MAX_THREADS_PER_BLOCK)
        except: # above won't work unless CUDA>=2.2
            pass
        self.block = (blocksize, 1, 1)
        self.grid = (int(ceil(float(self.N) / blocksize)), 1)
        self.kernel_func_kwds = {'block':self.block, 'grid':self.grid}
        mydtype = self.mydtype
        N = self.N
        eqs = self.eqs
        statevars_arr = gpuarray.to_gpu(array(self.G._S.flatten(), dtype=mydtype))
        self.I = gpuarray.to_gpu(array(I, dtype=mydtype))
        self.statevars_arr = statevars_arr
        self.I_offset = gpuarray.to_gpu(array(I_offset, dtype=int32))
        # SPIKES
        if self.criterion.type == 'spikes':
            self.initialize_spikes(spiketimes, spiketimes_offset)
        # TRACES
        if self.criterion.type == 'traces':
            self.initialize_traces(inputs, inputs_offset)
        self.criterion.initialize_cuda_variables()
        self.initialize_delays(spikedelays)
        self.initialize_refractory(refractory)
        self.initialize_kernel_arguments()

    def launch(self, duration, stepsize=1 * second):
        if stepsize is None:
            self.kernel_func(int32(0), int32(duration / self.dt),
                             *self.kernel_func_args, **self.kernel_func_kwds)
            pycuda.context.synchronize()
        else:
            stepsize = int(stepsize / self.dt)
            duration = int(duration / self.dt)
            for Tstart in xrange(0, duration, stepsize):
                Tend = Tstart + min(stepsize, duration - Tstart)
                self.kernel_func(int32(Tstart), int32(Tend),
                                 *self.kernel_func_args, **self.kernel_func_kwds)
                pycuda.context.synchronize()

    def get_coincidence_count(self):
        return self.num_coincidences.get()

    def get_spike_count(self):
        return self.spikecount.get()
    coincidence_count = property(fget=lambda self:self.get_coincidence_count())
    spike_count = property(fget=lambda self:self.get_spike_count())

