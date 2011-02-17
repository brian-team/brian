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

modelfitting_kernel_template = """
__global__ void runsim(
    int Tstart, int Tend,     // Start, end time as integer (t=T*dt)
    // State variables
    %SCALAR% *state_vars,     // State variables are offset from this
    double *I_arr,            // Input current
    int *I_arr_offset,        // Input current offset (for separate input
                              // currents for each neuron)
    int *spikecount,          // Number of spikes produced by each neuron
    int *num_coincidences,    // Count of coincidences for each neuron
    int *spiketimes,          // Array of all spike times as integers (begin and
                              // end each train with large negative value)
    int *spiketime_indices,   // Pointer into above array for each neuron
    int *spikedelay_arr,      // Integer delay for each spike
    int *refractory_arr,      // Integer refractory times
    int *next_allowed_spiketime_arr,// Integer time of the next allowed spike (for refractoriness)
    int onset                 // Time onset (only count spikes from here onwards)
    %COINCIDENCE_COUNT_DECLARE_EXTRA_STATE_VARIABLES%
    )
{
    const int neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_index>=%NUM_NEURONS%) return;
    %EXTRACT_STATE_VARIABLES%
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
    const int refractory = refractory_arr[neuron_index];
    int next_allowed_spiketime = next_allowed_spiketime_arr[neuron_index];
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
        const bool is_refractory = (T<=next_allowed_spiketime);
        const bool has_spiked = (%THRESHOLD%)&&!is_refractory;
        nspikes += has_spiked*(T>=onset);
        // Reset
        if(has_spiked||is_refractory)
        {
            %RESET%
        }
        if(has_spiked)
            next_allowed_spiketime = T+refractory;
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
    %COINCIDENCE_COUNT_STORE_VARIABLES%
    next_allowed_spiketime_arr[neuron_index] = next_allowed_spiketime;
    spiketime_indices[neuron_index] = spiketime_index;
    num_coincidences[neuron_index] = ncoinc;
    spikecount[neuron_index] = nspikes;
}
"""

coincidence_counting_algorithm_src = {
    'inclusive':{
        '%COINCIDENCE_COUNT_DECLARE_EXTRA_STATE_VARIABLES%':'',
        '%COINCIDENCE_COUNT_INIT%':'',
        '%COINCIDENCE_COUNT_TEST%':'''
            ncoinc += has_spiked && (((last_spike_time+%DELTA%)>=Tspike) || ((next_spike_time-%DELTA%)<=Tspike));
            ''',
        '%COINCIDENCE_COUNT_NEXT%':'',
        '%COINCIDENCE_COUNT_STORE_VARIABLES%':'',
        },
    'exclusive':{
        '%COINCIDENCE_COUNT_DECLARE_EXTRA_STATE_VARIABLES%':''',
            bool *last_spike_allowed_arr,
            bool *next_spike_allowed_arr
        ''',
        '%COINCIDENCE_COUNT_INIT%':'''
            bool last_spike_allowed = last_spike_allowed_arr[neuron_index];
            bool next_spike_allowed = next_spike_allowed_arr[neuron_index];
            ''',
        '%COINCIDENCE_COUNT_TEST%':'''
            bool near_last_spike = last_spike_time+%DELTA%>=Tspike;
            bool near_next_spike = next_spike_time-%DELTA%<=Tspike;
            near_last_spike = near_last_spike && has_spiked;
            near_next_spike = near_next_spike && has_spiked;
            ncoinc += (near_last_spike&&last_spike_allowed) || (near_next_spike&&next_spike_allowed);
            bool near_both_allowed = (near_last_spike&&last_spike_allowed) && (near_next_spike&&next_spike_allowed);
            last_spike_allowed = last_spike_allowed && !near_last_spike;
            next_spike_allowed = (next_spike_allowed && !near_next_spike) || near_both_allowed;
            ''',
        '%COINCIDENCE_COUNT_NEXT%':'''
            last_spike_allowed = next_spike_allowed;
            next_spike_allowed = true;
            ''',
        '%COINCIDENCE_COUNT_STORE_VARIABLES%':'''
            last_spike_allowed_arr[neuron_index] = last_spike_allowed;
            next_spike_allowed_arr[neuron_index] = next_spike_allowed;
            ''',
        },
    }

def generate_modelfitting_kernel_src(G, eqs, threshold, reset, dt, num_neurons,
                                     delta,
                                     coincidence_count_algorithm='exclusive',
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
    def __init__(self, G, eqs, I, I_offset, spiketimes, spiketimes_offset,
                       spikedelays, refractory,
                       delta, onset=0 * ms,
                       coincidence_count_algorithm='exclusive',
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
        self.coincidence_count_algorithm = coincidence_count_algorithm
        threshold = G._threshold
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
        self.threshold = threshold
        reset = G._resetfun
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
        self.reset = reset
        self.kernel_src = generate_modelfitting_kernel_src(
                  self.G, eqs, threshold, reset, dt, N, delta,
                  coincidence_count_algorithm=coincidence_count_algorithm,
                  precision=precision, scheme=scheme)

    def reinit_vars(self, I, I_offset, spiketimes, spiketimes_offset,
                    spikedelays, refractory):
        
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
        self.spiketimes = gpuarray.to_gpu(array(rint(spiketimes / self.dt), dtype=int32))
        self.spiketime_indices = gpuarray.to_gpu(array(spiketimes_offset, dtype=int32))
        self.num_coincidences = gpuarray.to_gpu(zeros(N, dtype=int32))
        self.spikecount = gpuarray.to_gpu(zeros(N, dtype=int32))
        self.spikedelay_arr = gpuarray.to_gpu(array(rint(spikedelays / self.dt), dtype=int32))
        if isinstance(refractory, float):
            refractory = refractory*ones(N)
        self.refractory_arr = gpuarray.to_gpu(array(rint(refractory / self.dt), dtype=int32))
        self.next_allowed_spiketime_arr = gpuarray.to_gpu(-ones(N, dtype=int32))
        self.next_spike_allowed_arr = gpuarray.to_gpu(ones(N, dtype=bool))
        self.last_spike_allowed_arr = gpuarray.to_gpu(zeros(N, dtype=bool))
        self.kernel_func_args = [self.statevars_arr,
                                 self.I,
                                 self.I_offset,
                                 self.spikecount,
                                 self.num_coincidences,
                                 self.spiketimes,
                                 self.spiketime_indices,
                                 self.spikedelay_arr,
                                 self.refractory_arr,
                                 self.next_allowed_spiketime_arr,
                                 int32(rint(self.onset / self.dt))]
        if self.coincidence_count_algorithm == 'exclusive':
            self.kernel_func_args += [self.last_spike_allowed_arr,
                                      self.next_spike_allowed_arr]

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

if __name__ == '__main__':
    import time
    from matplotlib.cm import jet
    from playdoh import *
    initialise_cuda()
    set_gpu_device(0)
    if 1:
        set_global_preferences(usecodegenthreshold=False)
        N = 10000
        nvar = 100
        delta = 4 * ms
        doplot = True
        eqs = '''
        dV/dt = (-V+R*I)/tau : volt
        Vt : volt
        tau : second
        R : ohm
        I : amp
        '''
        eqs += '\n'.join(['var'+str(i)+':0' for i in range(nvar)])
        eqs = Equations(eqs)
        Vr = 0.0 * volt
        Vt = 1.0 * volt
        I = loadtxt('../../../dev/ideas/cuda/modelfitting/current.txt')
        spiketimes = loadtxt('../../../dev/ideas/cuda/modelfitting/spikes.txt')
        spiketimes -= int(min(spiketimes))
        I_offset = zeros(N, dtype=int)
        spiketimes_offset = zeros(N, dtype=int)
        G = NeuronGroup(N, eqs, reset='V=0*volt; Vt+=1*volt', threshold='V>1*volt')
        G.R = rand(N) * 2e9 + 1e9
        G.tau = rand(N) * 49 * ms + 1 * ms
        spikedelays = rand(N) * 5 * ms
        duration = len(I) * G.clock.dt
        spiketimes = hstack((-1, spiketimes, float(duration) + 1))
        mf = GPUModelFitting(G, eqs, I, I_offset, spiketimes, spiketimes_offset,
                             spikedelays, 0*ms,
                             delta,
                             coincidence_count_algorithm='exclusive')
        mf.reinit_vars(I, I_offset, spiketimes, spiketimes_offset, spikedelays, 0*ms)
        print mf.kernel_src
        start_time = time.time()
        mf.launch(duration)
        running_time = time.time() - start_time

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
            scatter(R, tau, color=jet(spikecount / float(spikecount.max())))
            xlabel('R')
            ylabel('tau')
            title('Spike count, max = ' + str(spikecount.max()))
            axis('tight')

            subplot(222)
            scatter(R, tau, color=jet(num_coincidences / float(num_coincidences.max())))
            xlabel('R')
            ylabel('tau')
            title('Num coincidences, max = ' + str(num_coincidences.max()))
            axis('tight')

            spikecount -= num_coincidences
            num_coincidences -= spikecount
            num_coincidences[num_coincidences < 0] = 0
            maxcoinc = num_coincidences.max()
            num_coincidences = (1. * num_coincidences) / maxcoinc

            subplot(212)
            scatter(R, tau, color=jet(num_coincidences))
            xlabel('R')
            ylabel('tau')
            title('Hot = ' + str(maxcoinc) + ' excess coincidences, cool = 0 or less')
            axis('tight')

            show()
    if 0:
        N = 10000
        eqs = Equations('''
        dV/dt = (-V+R*I)/tau : volt
        dVt/dt = -(V-1*volt)/tau_t : volt
        R : ohm
        tau : second
        tau_t : second
        Vt_delta : volt
        I : amp
        ''')
        threshold = 'V>Vt'
        reset = '''
        V = 0*volt
        Vt += Vt_delta
        '''
        delta = 4 * ms
        src, declarations_seq = generate_modelfitting_kernel_src(eqs, threshold, reset,
                                               defaultclock.dt, N, delta)
        print src
        print
        print declarations_seq
    if 0:
        # test traces

        #clk = RegularClock(makedefaultclock=True)
        clk = defaultclock

        N = 1
        duration = 200 * ms
        delta = 4 * ms
        Ntarg = 20

        randspikes = hstack(([-1 * second], sort(rand(Ntarg) * duration * .9 + duration * 0.05), [duration + 1 * second]))
        #randspikes = sort(unique(array(randspikes/(2*delta), dtype=int)))*2*delta
        randspikes = sort(unique(array(randspikes / clk.dt, dtype=int))) * clk.dt + 1e-10

        eqs = Equations('''
        dV/dt = (-V+I)/(10*ms) : 1
        I : 1
        ''')
        threshold = 'V>1'
        reset = 'V=0'
        G = NeuronGroup(N, eqs, threshold=threshold, reset=reset, method='Euler',
                        clock=clk)
        #from brian.experimental.ccodegen import *
        #su = AutoCompiledNonlinearStateUpdater(eqs, G.clock, freeze=True)
        #G._state_updater = su
        #I = 1.1*ones(int(duration/defaultclock.dt))
        I = 5.0 * rand(int(duration / defaultclock.dt))
        #I = hstack((zeros(100), 10*ones(int(duration/defaultclock.dt))))
        #I = hstack((zeros(100), 10*ones(100))*(int(duration/defaultclock.dt)/200))
        #I = hstack((zeros(100), 10*exp(-linspace(0,2,100)))*(int(duration/defaultclock.dt)/200))
        #G.I = TimedArray(hstack((0, I)))
        #G.I = TimedArray(I[1:], clock=clk)
        G.I = TimedArray(I, clock=clk)
        M = StateMonitor(G, 'V', record=True, when='end', clock=clk)
        MS = SpikeMonitor(G)
        cc_ex = CoincidenceCounter(source=G, data=randspikes, delta=delta,
                                   coincidence_count_algorithm='exclusive')
        cc_in = CoincidenceCounter(source=G, data=randspikes, delta=delta,
                                   coincidence_count_algorithm='inclusive')
#        cc2 = CoincidenceCounter(source=G, data=randspikes[1:-1], delta=delta)
        run(duration)
        #spiketimes = array([-1*second, duration+1*second])
        #spiketimes_offset = zeros(N, dtype=int)
        #spiketimes = [-1*second]+MS[0]+[duration+1*second]
        spiketimes = randspikes
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
        oldnc = 0
        oldsc = 0
        allcoinc = []
        all_pst = []
        all_nst = []
        allspike = []
        all_nsa = []
        all_lsa = []

        if 0:
            for i in xrange(len(M.times)):
                mf.kernel_func(int32(i), int32(i + 1),
                                 *mf.kernel_func_args, **mf.kernel_func_kwds)
                #autoinit.context.synchronize()
                pycuda.context.synchronize()
                allV.append(mf.state_vars['V'].get())
                all_pst.append(mf.spiketimes.get()[mf.spiketime_indices.get()])
                all_nst.append(mf.spiketimes.get()[mf.spiketime_indices.get() + 1])
                all_nsa.append(mf.next_spike_allowed_arr.get()[0])
                all_lsa.append(mf.last_spike_allowed_arr.get()[0])
#        self.next_spike_allowed_arr = gpuarray.to_gpu(ones(N, dtype=bool))
#        self.last_spike_allowed_arr = gpuarray.to_gpu(zeros(N, dtype=bool))
                nc = mf.coincidence_count[0]
                if nc > oldnc:
                    oldnc = nc
                    allcoinc.append(i * clk.dt)
                sc = mf.spike_count[0]
                if sc > oldsc:
                    oldsc = sc
                    allspike.append(i * clk.dt)

        else:
            mf.launch(duration, stepsize=None)

        print 'Num target spikes:', len(randspikes) - 2
        print 'Predicted spike counts:', MS.nspikes, mf.spike_count[0]

        print 'Coincidences:'
        print 'GPU', mf.coincidence_count
#        print 'CPU bis inc', cc_in.coincidences
        print 'CPU bis exc', cc_ex.coincidences
#        print 'CPU', cc2.coincidences

#        for t in randspikes[1:-1]:
#            plot([t*second-delta, t*second+delta], [0, 0], lw=5, color=(.9,.9,.9))
#        plot(randspikes[1:-1], zeros(len(randspikes)-2), '+', ms=15)
#        plot(M.times, M[0])
        if len(allV):
#            plot(M.times, allV)
#            plot(allcoinc, zeros(len(allcoinc)), 'o')
#            figure()
            plot(M.times, array(all_pst) * clk.dt)
            plot(M.times, array(all_nst) * clk.dt)
            plot(randspikes[1:-1], randspikes[1:-1], 'o')
            plot(allspike, allspike, 'x')
            plot(allcoinc, allcoinc, '+')
            plot(M.times, array(all_nsa) * M.times, '--')
            plot(M.times, array(all_lsa) * M.times, '-.')
            predicted_spikes = allspike
            target_spikes = [t * second for t in randspikes]
            i = 0
            truecoinc = []
            for pred_t in predicted_spikes:
                 while target_spikes[i] < pred_t + delta:
                     if abs(target_spikes[i] - pred_t) < delta:
                         truecoinc.append((pred_t, target_spikes[i]))
                         i += 1
                         break
                     i += 1
            print 'Truecoinc:', len(truecoinc)
            for t1, t2 in truecoinc:
                plot([t1, t2], [t1, t2], ':', color=(0.5, 0, 0), lw=3)
#        show()
