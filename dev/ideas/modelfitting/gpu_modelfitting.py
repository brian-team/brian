from brian import Equations, NeuronGroup, Clock, CoincidenceCounter, Network, zeros, array, \
                    ones, kron, ms, second, concatenate, hstack, sort, nonzero, diff, TimedArray, \
                    reshape, sum, log, Monitor, NetworkOperation, defaultclock, linspace, vstack, \
                    arange, sort_spikes, rint, SpikeMonitor, Connection, Threshold, Reset, \
                    int32, double, VariableReset, StringReset, VariableThreshold, StringThreshold
from brian.tools.statistics import firing_rate, get_gamma_factor
from playdoh import *
import brian.optimiser as optimiser
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
                code += '        ' + line + '\n'
        return code




def get_cuda_template():
    return """
__global__ void runsim(
    // ITERATIONS
    int Tstart, int Tend,             // Start, end time as integer (t=T*dt)
    int duration,                     // Total duration as integer
    
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
    
    // DATA DECLARE
    %DATA_DECLARE%
    
    // STATEMONITOR DECLARE
    %STATEMONITOR_DECLARE%
    
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
    
    // DATA INIT
    %DATA_INIT%
    
    // STATEMONITOR INIT
    %STATEMONITOR_INIT%
    
    // CRITERION INITIALIZATION
    %CRITERION_INIT%
    
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
        
        // DATA UPDATE
        %DATA_UPDATE%
        
        // STATEMONITOR UPDATE
        %STATEMONITOR_UPDATE%
        
        // CRITERION TIMESTEP
        %CRITERION_TIMESTEP%
    }
    // STORE VARIABLES
    %STORE_VARIABLES%
    
    // STATEMONITOR END
    %STATEMONITOR_END%
    
    // CRITERION END
    %CRITERION_END%
    
    next_allowed_spiketime_arr[neuron_index] = next_allowed_spiketime;
    
    // END
    %DATA_END%
}
    """




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
    def __init__(self,
                 G,
                 eqs,
                 criterion, # Criterion object
                 input_var,
                 onset=0*ms,
                 precision=default_precision,
                 statemonitor_var=None,
                 duration=None,
                 scheme=euler_scheme
                 ):
        eqs.prepare()
        self.precision = precision
        self.scheme = scheme
        if precision == 'double':
            self.mydtype = float64
        else:
            self.mydtype = float32
        self.N = len(G)
        self.dt = G.clock.dt
        self.onset = onset
        self.eqs = eqs
        self.G = G
        self.duration = int(duration/self.dt)
        self.input_var = input_var
        self.statemonitor_var = statemonitor_var
        self.criterion = criterion
        self.generate_code()

    def generate_threshold_code(self, src):
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
        
        # Substitute threshold
        src = src.replace('%THRESHOLD%', threshold)
        
        return src
     
    def generate_reset_code(self, src):
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
#        self.reset = reset
        # Substitute reset
        reset = '\n            '.join(line.strip() + ';' for line in reset.split('\n') if line.strip())
        src = src.replace('%RESET%', reset)
        
        return src
            
    def generate_data_code(self, src):
        # Substitute spikes/traces declare
        if self.criterion.type == 'spikes':
            data_declare = """
    int *spiketimes,          // Array of all spike times as integers (begin and
                              // end each train with large negative value)
    int *spiketime_indices,   // Pointer into above array for each neuron
            """
        if self.criterion.type == 'traces':
            data_declare = """
    double *traces_arr,
    int *traces_arr_offset,
            """
        src = src.replace('%DATA_DECLARE%', data_declare)
        
        # Substitute spikes/traces init
        if self.criterion.type == 'spikes':
            data_init = """
    int spiketime_index = spiketime_indices[neuron_index];
            """
        if self.criterion.type == 'traces':
            data_init = """
    int trace_offset = traces_arr_offset[neuron_index];
    double trace_value = 0.0;
    int Tdelay = 0;
            """
        src = src.replace('%DATA_INIT%', data_init)
        
        # Substitute spikes/traces update
        if self.criterion.type == 'spikes':
            data_update = """
            """
        if self.criterion.type == 'traces':
            data_update = """
        Tdelay = T+spikedelay;
        if ((Tdelay>=0)&(Tdelay<duration-1)) {
            trace_value = traces_arr[Tdelay+trace_offset+1];
        }
            """
        src = src.replace('%DATA_UPDATE%', data_update)
        
        # Substitute spikes/traces end
        if self.criterion.type == 'spikes':
            data_end = """
            spiketime_indices[neuron_index] = spiketime_index;
            """
        if self.criterion.type == 'traces':
            data_end = """
            """
        src = src.replace('%DATA_END%', data_end)
        
        return src
    
    def generate_statemonitor_code(self, src):
        if self.statemonitor_var is not None:
            declare = """
    %SCALAR% *statemonitor_values,
    int *statemonitor_offsets,
            """
            init = """
    int statemonitor_offset = statemonitor_offsets[neuron_index];
            """
            update = """
        statemonitor_values[statemonitor_offset] = %s;
        statemonitor_offset++;
            """ % self.statemonitor_var
            end = """
    statemonitor_offsets[neuron_index] = statemonitor_offset;
            """
        else:
            declare = ""
            init = ""
            update = ""
            end = ""
        src = src.replace('%STATEMONITOR_DECLARE%', declare)
        src = src.replace('%STATEMONITOR_INIT%', init)
        src = src.replace('%STATEMONITOR_UPDATE%', update)
        src = src.replace('%STATEMONITOR_END%', end)
        return src
    
    def generate_code(self):
        self.eqs.prepare()
        src = get_cuda_template()
        # Substitute state variable declarations
        indexvar = dict((v, k) for k, v in self.G.var_index.iteritems() if isinstance(k, str) and k!='I')
        extractions = '\n    '.join('%SCALAR% *'+name+'_arr = state_vars+'+str(i*self.N)+';' for i, name in indexvar.iteritems())
        src = src.replace('%EXTRACT_STATE_VARIABLES%', extractions)
        
        # Substitute load variables
        loadvar_names = self.eqs._diffeq_names + []
        loadvar_names.remove('I') # I is assumed to be a parameter and loaded per time step
        loadvars = '\n    '.join('%SCALAR% ' + name + ' = ' + name + '_arr[neuron_index];' for name in loadvar_names)
        
        # simple equation patterns
        for name in self.eqs._string.keys():
            if name not in self.eqs._diffeq_names:
                loadvars += '\n    %SCALAR% ' + name + ' = 0;'
        
        src = src.replace('%LOAD_VARIABLES%', loadvars)
        
        # Substitute save variables
        savevars = '\n    '.join(name + '_arr[neuron_index] = ' + name + ';' for name in loadvar_names)
        src = src.replace('%STORE_VARIABLES%', savevars)
        
        src = self.generate_threshold_code(src)
        src = self.generate_reset_code(src)
        src = self.generate_data_code(src)
        src = self.generate_statemonitor_code(src)
        
        # Substitute state update
        sulines = ModelfittingGPUCodeGenerator(dtype=self.precision).generate(self.eqs, self.scheme)
        
        # simple equation patterns
        for name in self.eqs._string.keys():
            if name not in self.eqs._diffeq_names:
                sulines += '        ' + name + ' = ' + self.eqs._string[name] + ';\n'
        
        sulines = re.sub(r'\bdt\b', '%DT%', sulines)
        src = src.replace('%STATE_UPDATE%', sulines.strip())
        
        # Substitute criterion code
        criterion_code = self.criterion.get_cuda_code()
        for search, replace in criterion_code.iteritems():
            src = src.replace(search, replace)
        
        # Substitute dt
        src = src.replace('%DT%', str(float(self.dt)))
        # Substitute SCALAR
        src = src.replace('%SCALAR%', self.precision)
        # Substitute number of neurons
        src = src.replace('%NUM_NEURONS%', str(self.N))
        # Substitute input var name
        src = src.replace('${input_var}', str(self.input_var))
        
        self.kernel_src = src
#        log_info(src)

    def initialize_spikes(self, spiketimes, spiketimes_indices):
        self.spiketimes = gpuarray.to_gpu(array(rint(spiketimes / self.dt), dtype=int32))
        self.spiketime_indices = gpuarray.to_gpu(array(spiketimes_indices, dtype=int32))
        
    def initialize_traces(self, traces, traces_offset):
        self.traces = gpuarray.to_gpu(array(traces, dtype=double))
        self.traces_offset = gpuarray.to_gpu(array(traces_offset, dtype=int32))

    def initialize_delays(self, spikedelays):
        self.spikedelay_arr = gpuarray.to_gpu(array(rint(spikedelays / self.dt), dtype=int32))
    
    def initialize_refractory(self, refractory):
        if isinstance(refractory, float):
            refractory = refractory*ones(self.N)
        self.refractory_arr = gpuarray.to_gpu(array(rint(refractory / self.dt), dtype=int32))
        self.next_allowed_spiketime_arr = gpuarray.to_gpu(-ones(self.N, dtype=int32))

    def initialize_statemonitor(self):
        self.statemonitor_values = gpuarray.to_gpu(zeros(self.N*self.duration, dtype=self.precision))
        self.statemonitor_offsets = gpuarray.to_gpu(arange(0, self.duration*self.N+1,
                                                           self.duration, 
                                                           dtype=int32))
    
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
            self.kernel_func_args += [self.traces,
                                      self.traces_offset]
        if self.statemonitor_var is not None:
            self.kernel_func_args += [self.statemonitor_values,
                                      self.statemonitor_offsets]
        
        self.kernel_func_args += [int32(rint(self.onset / self.dt))]

    def reinit_vars(self, criterion,
                    I, I_offset,
                    spiketimes = None, spiketimes_offset = None,
                    traces = None, traces_offset = None,
                    spikedelays = None, refractory = None):
        self.criterion = criterion
        
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
            self.initialize_traces(traces, traces_offset)
        self.criterion.initialize_cuda_variables()
        self.initialize_delays(spikedelays)
        self.initialize_refractory(refractory)
        
        if self.statemonitor_var is not None:
            self.initialize_statemonitor()
        
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
                self.kernel_func(int32(Tstart), int32(Tend), int32(duration),
                                 *self.kernel_func_args, **self.kernel_func_kwds)
                pycuda.context.synchronize()

    def get_statemonitor_values(self):
        values = self.statemonitor_values.get()
        values = values.reshape((self.N, -1))
        return values



