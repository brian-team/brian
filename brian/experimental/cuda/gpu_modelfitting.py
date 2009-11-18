from brian import *
import brian.optimiser as optimiser
import pycuda.autoinit as autoinit
import pycuda.driver as drv
import pycuda
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
try:
    from pycuda.compiler import SourceModule
except ImportError:
    from pycuda.driver import SourceModule
from numpy import *

__all__ = ['GPUModelFitting']

if drv.get_version()==(2,0,0): # cuda version
    default_precision = 'float'
elif drv.get_version()>(2,0,0):
    default_precision = 'double'
else:
    raise Exception,"CUDA 2.0 required"

modelfitting_kernel_template = """
__global__ void runsim(
    int Tstart, int Tend,     // Start, end time as integer (t=T*dt)
    // State variables
    %FUNC_DECLARE_STATE_VARIABLES%
    int *I_arr_offset,        // Input current offset (for separate input
                              // currents for each neuron)
    int *spikecount,          // Number of spikes produced by each neuron
    int *num_coincidences,    // Count of coincidences for each neuron
    int *spiketimes,          // Array of all spike times as integers (begin and
                              // end each train with large negative value)
    int *spiketime_indices,   // Pointer into above array for each neuron
    int *spikedelay_arr       // Integer delay for each spike
    )
{
    const int neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_index>=%NUM_NEURONS%) return;
    // Load variables at start
    %LOAD_VARIABLES%
    int spiketime_index = spiketime_indices[neuron_index];
    int last_spike_time = spiketimes[spiketime_index];
    int next_spike_time = spiketimes[spiketime_index+1];
    %COINCIDENCE_COUNT_INIT%
    int ncoinc = num_coincidences[neuron_index];
    int nspikes = spikecount[neuron_index];
    int I_offset = I_arr_offset[neuron_index];
    int spikedelay = spikedelay_arr[neuron_index];
    for(int T=Tstart; T<Tend; T++)
    {
        %SCALAR% t = T*%DT%;
        // Read input current
        %SCALAR% I = I_arr[T+I_offset];
                             // this is a global read for each thread, can maybe
                             // reduce this by having just one read per block,
                             // put it into shared memory, and then have all the
                             // threads in that block read it, we could even
                             // maybe buffer reads of I into shared memory -
                             // experiment with this 
        // State update
        %STATE_UPDATE%
        // Threshold
        const bool has_spiked = %THRESHOLD%;
        nspikes += has_spiked;
        // Reset
        if(has_spiked)
        {
            %RESET%
        }
        // Coincidence counter
        const int Tspike = T+spikedelay;
        %COINCIDENCE_COUNT_TEST%
        if(Tspike>=next_spike_time){
            spiketime_index++;
            last_spike_time = next_spike_time;
            next_spike_time = spiketimes[spiketime_index+1];
            %COINCIDENCE_COUNT_NEXT%
        }
    }
    // Store variables at end
    %STORE_VARIABLES%
    spiketime_indices[neuron_index] = spiketime_index;
    num_coincidences[neuron_index] = ncoinc;
    spikecount[neuron_index] = nspikes;
}
"""

coincidence_counting_algorithm_src = {
    'inclusive':{
        '%COINCIDENCE_COUNT_INIT%':'',
        '%COINCIDENCE_COUNT_TEST%':'''
            ncoinc += has_spiked && (((last_spike_time+%DELTA%)>=Tspike) || ((next_spike_time-%DELTA%)<=Tspike));
            ''',
        '%COINCIDENCE_COUNT_NEXT%':''
        },
    'exclusive':{
        '%COINCIDENCE_COUNT_INIT%':'''
            bool last_spike_allowed = true, next_spike_allowed = true;
            ''',
        '%COINCIDENCE_COUNT_TEST%':'''
            ncoinc += has_spiked &&
                      ((((last_spike_time+%DELTA%)>=Tspike)&&last_spike_allowed)
                       ||
                       (((next_spike_time-%DELTA%)<=Tspike)&&next_spike_allowed));
            last_spike_allowed = !(has_spiked && ((last_spike_time+%DELTA%)>=Tspike));
            next_spike_allowed = !(has_spiked && (((last_spike_time+%DELTA%)<Tspike) && ((next_spike_time-%DELTA%)<=Tspike)));
            ''',
        '%COINCIDENCE_COUNT_NEXT%':'''
            last_spike_allowed = next_spike_allowed;
            next_spike_allowed = true;
            '''
        },
    }

def generate_modelfitting_kernel_src(eqs, threshold, reset, dt, num_neurons,
                                     delta,
                                     coincidence_count_algorithm='exclusive',
                                     precision=default_precision):
    eqs.prepare()
    src = modelfitting_kernel_template
    # Substitute state variable declarations
    declarations = '\n    '.join('%SCALAR% *'+name+'_arr,' for name in eqs._diffeq_names)
    declarations_seq = eqs._diffeq_names+[]
    src = src.replace('%FUNC_DECLARE_STATE_VARIABLES%', declarations)
    # Substitute load variables
    loadvar_names = eqs._diffeq_names+[]
    loadvar_names.remove('I') # I is assumed to be a parameter and loaded per time step
    loadvars = '\n    '.join('%SCALAR% '+name+' = '+name+'_arr[neuron_index];' for name in loadvar_names)
    src = src.replace('%LOAD_VARIABLES%', loadvars)
    # Substitute save variables
    savevars = '\n    '.join(name+'_arr[neuron_index] = '+name+';' for name in loadvar_names)
    src = src.replace('%STORE_VARIABLES%', savevars)
    # Substitute threshold
    src = src.replace('%THRESHOLD%', threshold)
    # Substitute reset
    reset = '\n            '.join(line.strip()+';' for line in reset.split('\n') if line.strip())
    src = src.replace('%RESET%', reset)
    # Substitute state update
    sulines = ''
    all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
    for name in eqs._diffeq_names:
        namespace = eqs._namespace[name]
        expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
        if name in eqs._diffeq_names_nonzero:
            sulines += '        %SCALAR% '+name+'__tmp = '+expr+';\n'
    for name in eqs._diffeq_names_nonzero:
        sulines += '        '+name+' += %DT%*'+name+'__tmp;\n'
    src = src.replace('%STATE_UPDATE%', sulines.strip())
    # Substitute coincidence counting algorithm
    ccalgo = coincidence_counting_algorithm_src[coincidence_count_algorithm]
    for search, replace in ccalgo.iteritems():
        src = src.replace(search, replace)
    # Substitute dt
    src = src.replace('%DT%', str(float(dt)))
    # Substitute SCALAR
    src = src.replace('%SCALAR%', precision)
    # Substitute number of neurons
    src = src.replace('%NUM_NEURONS%', str(num_neurons))
    # Substitute delta, the coincidence window half-width
    src = src.replace('%DELTA%', str(int(delta/dt)))
    return src, declarations_seq

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
    ``launch(duration)``
        Runs the kernel on the GPU for simulation time duration.
    
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
    def __init__(self, G, eqs, I, I_offset, spiketimes, spiketimes_offset, spikedelays,
                       delta, coincidence_count_algorithm='exclusive',
                       precision=default_precision):
        eqs.prepare()
        self.precision = precision
        if precision=='double':
            self.mydtype = float64
        else:
            self.mydtype = float32
        self.N = N = len(G)
        self.dt = dt = G.clock.dt
        self.delta = delta
        self.eqs = eqs
        self.G = G
        threshold = G._threshold
        if threshold.__class__ is Threshold:
            state = threshold.state
            if isinstance(state, int):
                state = eqs._diffeq_names[state]            
            threshold = state+'>'+str(float(threshold.threshold))
        elif isinstance(threshold, VariableThreshold):
            state = threshold.state
            if isinstance(state, int):
                state = eqs._diffeq_names[state]            
            threshold = state+'>'+threshold.threshold_state
        elif isinstance(threshold, StringThreshold):
            namespace = threshold._namespace
            expr = threshold._expr
            all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
            expr = optimiser.freeze(expr, all_variables, namespace)
            threshold = expr
        else:
            raise ValueError('Threshold must be constant, VariableThreshold or StringThreshold.')
        self.threshold = threshold
        reset = G._resetfun
        if reset.__class__ is Reset:
            state = reset.state
            if isinstance(state, int):
                state = eqs._diffeq_names[state]
            reset = state+' = '+str(float(reset.resetvalue))
        elif isinstance(reset, VariableReset):
            state = reset.state
            if isinstance(state, int):
                state = eqs._diffeq_names[state]
            reset = state+' = '+reset.resetvaluestate
        elif isinstance(reset, StringReset):
            namespace = reset._namespace
            expr = reset._expr
            all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
            expr = optimiser.freeze(expr, all_variables, namespace)
            reset = expr
        self.reset = reset
        self.kernel_src, self.declarations_seq = generate_modelfitting_kernel_src(eqs, threshold, reset, dt, N, delta,
                                                                                  coincidence_count_algorithm=coincidence_count_algorithm,
                                                                                  precision=precision)
        print self.kernel_src
        self.kernel_module = SourceModule(self.kernel_src)
        self.kernel_func = self.kernel_module.get_function('runsim')
        self.reinit_vars(I, I_offset, spiketimes, spiketimes_offset, spikedelays)
        # TODO: compute block, grid, etc. with best maximum blocksize
        blocksize = 256
        self.block = (blocksize, 1, 1)
        self.grid = (int(ceil(float(N)/blocksize)), 1)
        self.kernel_func_kwds = {'block':self.block, 'grid':self.grid}
            
    def reinit_vars(self, I, I_offset, spiketimes, spiketimes_offset, spikedelays):
        mydtype = self.mydtype
        N = self.N
        eqs = self.eqs
        statevar_names = eqs._diffeq_names+[]
        statevar_names.remove('I')
        self.state_vars = dict((name, self.G.state_(name)) for name in statevar_names)
        for name, val in self.state_vars.items():
            self.state_vars[name] = gpuarray.to_gpu(array(val, dtype=mydtype))
        self.I = gpuarray.to_gpu(array(I, dtype=mydtype))
        self.state_vars['I'] = self.I
        self.I_offset = gpuarray.to_gpu(array(I_offset, dtype=int))
        self.spiketimes = gpuarray.to_gpu(array(spiketimes/self.dt, dtype=int))
        self.spiketime_indices = gpuarray.to_gpu(array(spiketimes_offset, dtype=int))
        self.num_coincidences = gpuarray.to_gpu(zeros(N, dtype=int))
        self.spikecount = gpuarray.to_gpu(zeros(N, dtype=int))
        self.spikedelay_arr = gpuarray.to_gpu(array(spikedelays/self.dt, dtype=int))
        self.kernel_func_args = [self.state_vars[name] for name in self.declarations_seq]
        self.kernel_func_args += [self.I_offset, self.spikecount, self.num_coincidences, self.spiketimes, self.spiketime_indices, self.spikedelay_arr]
        
    def launch(self, duration, stepsize=None):
        if stepsize is None:
            self.kernel_func(int32(0), int32(duration/self.dt),
                             *self.kernel_func_args, **self.kernel_func_kwds)
            autoinit.context.synchronize()
        else:
            stepsize = int(stepsize/self.dt)
            duration = int(duration/self.dt)
            for Tstart in xrange(0, duration, stepsize):
                Tend = Tstart+min(Tstart, duration-Tstart)
                self.kernel_func(int32(Tstart), int32(Tend),
                                 *self.kernel_func_args, **self.kernel_func_kwds)
                autoinit.context.synchronize()
                
    
    def get_coincidence_count(self):
        return self.num_coincidences.get()
    def get_spike_count(self):
        return self.spikecount.get()
    coincidence_count = property(fget=lambda self:self.get_coincidence_count())
    spike_count = property(fget=lambda self:self.get_spike_count())

if __name__=='__main__':
    import time
    from matplotlib.cm import jet
    if 0:
        N = 10000
        delta = 4*ms
        doplot = True
        eqs = Equations('''
        dV/dt = (-V+R*I)/tau : 1
        tau : second
        R : 1
        I : 1
        ''')
        Vr = 0.0
        Vt = 1.0
        I = loadtxt('../../../dev/ideas/cuda/modelfitting/current.txt')
        spiketimes = loadtxt('../../../dev/ideas/cuda/modelfitting/spikes.txt')
        spiketimes -= int(min(spiketimes))
        I_offset = zeros(N, dtype=int)
        spiketimes_offset = zeros(N, dtype=int)
        G = NeuronGroup(N, eqs, reset='V=Vr', threshold='V>Vt')
        G.R = rand(N)*2e9+1e9
        G.tau = rand(N)*49*ms+1*ms
        spikedelays = rand(N)*5*ms
        duration = len(I)*G.clock.dt
        spiketimes = hstack((-1, spiketimes, float(duration)+1))
        mf = GPUModelFitting(G, eqs, I, I_offset, spiketimes, spiketimes_offset,
                             spikedelays,
                             delta,
                             coincidence_count_algorithm='exclusive')
        print mf.kernel_src
        start_time = time.time()
        mf.launch(duration)
        running_time = time.time()-start_time
        
        print 'N:', N
        print 'Duration:', duration
        print 'Total running time:', running_time

        if doplot:
            
            spikecount = mf.spike_count
            num_coincidences = mf.coincidence_count
            R = G.R
            tau = G.tau
            
            print 'Spike count varies between', spikecount.min(), 'and', spikecount.max()
            print 'Num coincidences varies between', num_coincidences.min(), 'and', num_coincidences.max()
    
            subplot(221)
            scatter(R, tau, color=jet(spikecount/float(spikecount.max())))
            xlabel('R')
            ylabel('tau')
            title('Spike count, max = '+str(spikecount.max()))
            axis('tight')
        
            subplot(222)
            scatter(R, tau, color=jet(num_coincidences/float(num_coincidences.max())))
            xlabel('R')
            ylabel('tau')
            title('Num coincidences, max = '+str(num_coincidences.max()))
            axis('tight')
            
            spikecount -= num_coincidences
            num_coincidences -= spikecount
            num_coincidences[num_coincidences<0] = 0
            maxcoinc = num_coincidences.max()
            num_coincidences = (1.*num_coincidences)/maxcoinc
            
            subplot(212)
            scatter(R, tau, color=jet(num_coincidences))
            xlabel('R')
            ylabel('tau')
            title('Hot = '+str(maxcoinc)+' excess coincidences, cool = 0 or less')
            axis('tight')
        
            show()        
    if 0:
        N = 10000
        eqs = Equations('''
        dV/dt = (-V+R*I)/tau : 1
        dVt/dt = -(V-1)/tau_t : 1
        R : 1
        tau : second
        tau_t : second
        Vt_delta : 1
        I : 1
        ''')
        threshold = 'V>Vt'
        reset = '''
        V = 0
        Vt += Vt_delta
        '''
        delta = 4*ms
        src, declarations_seq =  generate_modelfitting_kernel_src(eqs, threshold, reset,
                                               defaultclock.dt, N, delta)
        print src
        print
        print declarations_seq
    if 1:
        # test traces
                    
        #clk = RegularClock(makedefaultclock=True)
        clk = defaultclock
        
        N = 1
        duration = 100*ms
        eqs = Equations('''
        dV/dt = (-V+I)/(10*ms) : 1
        I : 1
        ''')
        threshold = 'V>1'
        reset = 'V=0'
        G = NeuronGroup(N, eqs, threshold=threshold, reset=reset, method='Euler',
                        clock=clk)
        from brian.experimental.ccodegen import *
        su = AutoCompiledNonlinearStateUpdater(eqs, G.clock, freeze=True)
        G._state_updater = su
        #I = 1.1*ones(int(duration/defaultclock.dt))
        I = 3.0*rand(int(duration/defaultclock.dt))
        #I = hstack((zeros(100), 10*ones(int(duration/defaultclock.dt))))
        #I = hstack((zeros(100), 10*ones(100))*(int(duration/defaultclock.dt)/200))
        #I = hstack((zeros(100), 10*exp(-linspace(0,2,100)))*(int(duration/defaultclock.dt)/200))
        #G.I = TimedArray(hstack((0, I)))
        #G.I = TimedArray(I[1:], clock=clk)
        G.I = TimedArray(I, clock=clk)
        M = StateMonitor(G, 'V', record=True, when='end', clock=clk)
        MS = SpikeMonitor(G)
        run(duration)
        delta = 4*ms
        #spiketimes = array([-1*second, duration+1*second])
        #spiketimes_offset = zeros(N, dtype=int)
        spiketimes = [-1*second]+MS[0]+[duration+1*second]
        spiketimes_offset = zeros(N, dtype=int)
        #I = array([0]+I)
        I_offset = zeros(N, dtype=int)
        spikedelays = zeros(N)
        reinit_default_clock()
        G.V = 0
        mf = GPUModelFitting(G, eqs, I, I_offset, spiketimes, spiketimes_offset,
                             spikedelays,
                             delta,
                             coincidence_count_algorithm='exclusive')
        allV = []
        for i in xrange(int(duration/defaultclock.dt)):
            mf.kernel_func(int32(i), int32(i+1),
                             *mf.kernel_func_args, **mf.kernel_func_kwds)
            autoinit.context.synchronize()
            allV.append(mf.state_vars['V'].get())
        
        print mf.coincidence_count    
        
        plot(M[0])
        plot(allV)
        show()