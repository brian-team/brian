from brian import Equations, NeuronGroup, Clock, CoincidenceCounter, Network, zeros, array, \
                    ones, kron, ms, second, concatenate, hstack, sort, nonzero, diff, TimedArray, \
                    reshape, sum, log, Monitor, NetworkOperation, defaultclock, linspace, vstack, \
                    arange, sort_spikes, rint, SpikeMonitor, Connection, StateMonitor
from brian.tools.statistics import firing_rate, get_gamma_factor
try:
    from playdoh import *
except Exception, e:
    print e
    raise ImportError("Playdoh must be installed (https://code.google.com/p/playdoh/)")

try:
    import pycuda
    from gpu_modelfitting import GPUModelFitting
    can_use_gpu = True
except ImportError:
    can_use_gpu = False
from brian.experimental.codegen.integration_schemes import *
from criteria import *
import sys, cPickle




class DataTransformer(object):
    """
    Transform spike, input and trace data from user-friendly data structures,
    like 2 dimensional arrays or lists of spikes, into algorithm- and GPU-friendly structures,
    i.e. inline vectors (1 dimensional) easily parallelizable.
    """
    def __init__(self, neurons, input, spikes = None, traces = None, dt = defaultclock.dt,
                 slices = 1, overlap = 0*ms, groups = 1):
        self.neurons = neurons # number of particles on the node
        self.input = input # a IxT array
        self.spikes = spikes # a list of spikes [(i,t)...]
        self.traces = traces # a KxT array
        self.slices = slices
        self.overlap = overlap
        self.groups = groups
        self.dt = dt
        # ensure 2 dimensions
        if self.input.ndim == 1:
            self.input = self.input.reshape((1,-1))
        if self.traces is not None:
            if self.traces.ndim == 1:
                self.traces = self.traces.reshape((1,-1))
        self.inputs_count = self.input.shape[0]
        self.T = self.input.shape[1] # number of steps
        self.duration = self.T*self.dt
        self.subpopsize = self.neurons/self.groups # number of neurons per group: nodesize/groups
        self.input = self.input[:,0:self.slices * (self.T / self.slices)] # makes sure that len(input) is a multiple of slices
        self.sliced_steps = self.T / self.slices # timesteps per slice
        self.overlap_steps = int(self.overlap / self.dt) # timesteps during the overlap
        self.total_steps = self.sliced_steps + self.overlap_steps # total number of timesteps
        self.sliced_duration = self.overlap + self.duration / self.slices # duration of the vectorized simulation
        self.N = self.neurons * self.slices # TOTAL number of neurons on this node
        self.input = hstack((zeros((self.inputs_count, self.overlap_steps)), self.input)) # add zeros at the beginning because there is no overlap from the previous slice
        
    def slice_spikes(self, spikes):
        # from standard structure to standard structure
        sliced_spikes = []
        slice_length = self.sliced_steps*self.dt
        for (i,t) in spikes:
            slice = int(t/slice_length)
            newt = self.overlap + (t % slice_length)*second
            newi = i + self.groups*slice
            # discard unreachable spikes
            if newi >= (self.slices*self.groups): continue
            sliced_spikes.append((newi, newt)) 
        sliced_spikes = sort_spikes(sliced_spikes)
        return sliced_spikes

    def slice_traces(self, traces):
        # from standard structure to standard structure
        k = traces.shape[0]
        sliced_traces = zeros((k*self.slices, self.total_steps))
        for slice in xrange(self.slices):
            i0 = slice*k
            i1 = (slice+1)*k
            j0 = slice*self.sliced_steps
            j1 = (slice+1)*self.sliced_steps
            sliced_traces[i0:i1,self.overlap_steps:] = traces[:,j0:j1]
            if slice>0:
                sliced_traces[i0:i1,:self.overlap_steps] = traces[:,j0-self.overlap_steps:j0]
        return sliced_traces

    def transform_spikes(self, spikes):
        # from standard structure to inline structure
        i, t = zip(*spikes)
        i = array(i)
        t = array(t)
        alls = []
        n = 0
        pointers = []
        model_target = []
        for j in xrange(self.groups):
            s = sort(t[i == j])
            s = hstack((-1 * second, s, self.duration + 1 * second))
            model_target.extend([j] * self.subpopsize)
            alls.append(s)
            pointers.append(n)
            n += len(s)
        pointers = array(pointers, dtype=int)
        model_target = array(hstack(model_target), dtype=int)
        spikes_inline = hstack(alls)
        spikes_offset = pointers[model_target]
        return spikes_inline, spikes_offset
    
    def transform_traces(self, traces):
        # from standard structure to inline structure
        K, T = traces.shape
        traces_inline = traces.flatten()
        traces_offset = array(kron(arange(K), T*ones(self.subpopsize)), dtype=int)
        return traces_inline, traces_offset




class Simulator(object):
    def __init__(self, model, reset, threshold, 
                 inputs, input_var = 'I', dt = defaultclock.dt,
                 refractory = 0*ms, max_refractory = None,
                 spikes = None, traces = None,
                 groups = 1,
                 slices = 1, overlap = 0*second,
                 onset = 0*second,
                 neurons = 1000, # = nodesize = number of neurons on this node = total number of neurons/slices
                 initial_values = None,
                 unit_type = 'CPU',
                 stepsize = 100*ms,
                 precision = 'double',
                 criterion = None,
                 statemonitor_var=None,
                 method = 'Euler'
                 ):
        self.model = model
        self.reset = reset
        self.threshold = threshold
        self.inputs = inputs
        self.input_var = input_var
        self.dt = dt
        self.refractory = refractory
        self.max_refractory = max_refractory
        self.spikes = spikes
        self.traces = traces
        self.initial_values = initial_values 
        self.groups = groups
        self.slices = slices
        self.overlap = overlap
        self.onset = onset
        self.neurons = neurons
        self.unit_type = unit_type
        self.statemonitor_var = statemonitor_var
        self.stepsize = stepsize
        self.precision = precision
        self.criterion = criterion
        self.method = method
        self.use_gpu = self.unit_type=='GPU'
        
        if self.statemonitor_var is not None:
            self.statemonitor_values = zeros(self.neurons)
        
        self.initialize_neurongroup()
        self.transform_data()
        self.inject_input()
        self.initialize_criterion(delays=zeros(self.neurons))
        
        if self.use_gpu:
            self.initialize_gpu()
            
    def initialize_neurongroup(self):
        # Add 'refractory' parameter on the CPU only
        if not self.use_gpu:
            if self.max_refractory is not None:
                refractory = 'refractory'
                self.model.add_param('refractory', second)
            else:
                refractory = self.refractory
        else:
            if self.max_refractory is not None:
                refractory = 0*ms
            else:
                refractory = self.refractory
        
        # Must recompile the Equations : the functions are not transfered after pickling/unpickling
        self.model.compile_functions()
        
        self.group = NeuronGroup(self.neurons,
                                 model=self.model,
                                 reset=self.reset,
                                 threshold=self.threshold,
                                 refractory=refractory,
                                 max_refractory = self.max_refractory,
                                 method = self.method,
                                 clock=Clock(dt=self.dt))
        
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value
    
    def initialize_gpu(self):
            # Select integration scheme according to method
            if self.method == 'Euler': scheme = euler_scheme
            elif self.method == 'RK': scheme = rk2_scheme
            elif self.method == 'exponential_Euler': scheme = exp_euler_scheme
            else: raise Exception("The numerical integration method is not valid")
            
            self.mf = GPUModelFitting(self.group, self.model, self.criterion_object,
                                      self.input_var,
                                      self.onset, 
                                      statemonitor_var = self.statemonitor_var,
                                      duration = self.sliced_duration,
                                      precision=self.precision, scheme=scheme)
    
    def transform_data(self):
        self.transformer = DataTransformer(self.neurons,
                                           self.inputs,
                                           spikes = self.spikes, 
                                           traces = self.traces,
                                           dt = self.dt,
                                           slices = self.slices,
                                           overlap = self.overlap, 
                                           groups = self.groups)
        self.total_steps = self.transformer.total_steps
        self.sliced_duration = self.transformer.sliced_duration
        
        self.sliced_inputs = self.transformer.slice_traces(self.inputs)
        self.inputs_inline, self.inputs_offset = self.transformer.transform_traces(self.sliced_inputs)
        
        if self.traces is not None:
            self.sliced_traces = self.transformer.slice_traces(self.traces)
            self.traces_inline, self.traces_offset = self.transformer.transform_traces(self.sliced_traces)
        else:
            self.sliced_traces, self.traces_inline, self.traces_offset = None, None, None
        
        if self.spikes is not None:
            self.sliced_spikes = self.transformer.slice_spikes(self.spikes)
            self.spikes_inline, self.spikes_offset = self.transformer.transform_spikes(self.sliced_spikes)
        else:
            self.sliced_spikes, self.spikes_inline, self.spikes_offset = None, None, None
    
    def inject_input(self):
        # Injects current in consecutive subgroups, where I_offset have the same value
        # on successive intervals
        I_offset = self.inputs_offset
        k = -1
        for i in hstack((nonzero(diff(I_offset))[0], len(I_offset) - 1)):
            I_offset_subgroup_value = I_offset[i]
            I_offset_subgroup_length = i - k
            sliced_subgroup = self.group.subgroup(I_offset_subgroup_length)
            input_sliced_values = self.inputs_inline[I_offset_subgroup_value:I_offset_subgroup_value + self.total_steps]
            sliced_subgroup.set_var_by_array(self.input_var, TimedArray(input_sliced_values, clock=self.group.clock))
            k = i
    
    def initialize_criterion(self, **criterion_params):
        # general criterion parameters
        params = dict(group=self.group, traces=self.sliced_traces, spikes=self.sliced_spikes, 
                      targets_count=self.groups*self.slices, duration=self.sliced_duration, onset=self.onset, 
                      spikes_inline=self.spikes_inline, spikes_offset=self.spikes_offset,
                      traces_inline=self.traces_inline, traces_offset=self.traces_offset)
        for key,val in criterion_params.iteritems():
            params[key] = val
        criterion_name = self.criterion.__class__.__name__
        
        # criterion-specific parameters
        if criterion_name == 'GammaFactor':
            params['delta'] = self.criterion.delta
            params['coincidence_count_algorithm'] = self.criterion.coincidence_count_algorithm
            self.criterion_object = GammaFactorCriterion(**params)
            
        if criterion_name == 'LpError':
            params['p'] = self.criterion.p
            params['varname'] = self.criterion.varname
            self.criterion_object = LpErrorCriterion(**params)
    
    def update_neurongroup(self, **param_values):
        """
        Inject fitting parameters into the NeuronGroup
        """
        # Sets the parameter values in the NeuronGroup object
        self.group.reinit()
        for param, value in param_values.iteritems():
            self.group.state(param)[:] = kron(value, ones(self.slices)) # kron param_values if slicing
        
        # Reinitializes the model variables
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value
    
    def combine_sliced_values(self, values):
        if type(values) is tuple:
            combined_values = tuple([sum(reshape(v, (self.slices, -1)), axis=0) for v in values])
        else:
            combined_values = sum(reshape(values, (self.slices, -1)), axis=0)
        return combined_values
    
    def run(self, **param_values):
        delays = param_values.pop('delays', zeros(self.neurons))
        refractory = param_values.pop('refractory', zeros(self.neurons))
        
        self.update_neurongroup(**param_values)

        # repeat spike delays and refractory to take slices into account
        delays = kron(delays, ones(self.slices))
        refractory = kron(refractory, ones(self.slices))
        
        # TODO: add here parameters to criterion_params if a criterion must use some parameters
        criterion_params = dict(delays=delays)

        self.update_neurongroup(**param_values)
        self.initialize_criterion(**criterion_params)
        
        if self.use_gpu:
            # Reinitializes the simulation object
            self.mf.reinit_vars(self.criterion_object,
                                self.inputs_inline, self.inputs_offset,
                                self.spikes_inline, self.spikes_offset,
                                self.traces_inline, self.traces_offset,
                                delays, refractory
                                )
            # LAUNCHES the simulation on the GPU
            self.mf.launch(self.sliced_duration, self.stepsize)
            # Synchronize the GPU values with a call to gpuarray.get()
            self.criterion_object.update_gpu_values()
        else:
            # set the refractory period
            if self.max_refractory is not None:
                self.group.refractory = refractory
            # Launch the simulation on the CPU
            self.group.clock.reinit()
            net = Network(self.group, self.criterion_object)
            if self.statemonitor_var is not None:
                self.statemonitor = StateMonitor(self.group, self.statemonitor_var, record=True)
                net.add(self.statemonitor)
            net.run(self.sliced_duration)
        
        sliced_values = self.criterion_object.get_values()
        combined_values = self.combine_sliced_values(sliced_values)
        values = self.criterion_object.normalize(combined_values)
        return values

    def get_statemonitor_values(self):
        if not self.use_gpu:
            return self.statemonitor.values
        else:
            return self.mf.get_statemonitor_values()




def simulate( model, reset = None, threshold = None, 
                 input = None, input_var = 'I', dt = defaultclock.dt,
                 refractory = 0*ms, max_refractory = None,
                 data = None,
                 groups = 1,
                 slices = 1,
                 overlap = 0*second,
                 neurons = 1,
                 initial_values = None,
                 use_gpu = False,
                 stepsize = 100*ms,
                 precision = 'double',
                 criterion = None,
                 record = None,
                 method = 'Euler',
                 **params):
    
    unit_type = 'CPU'
    if use_gpu: unit_type = 'GPU'
    
    for param in params.keys():
        if (param not in model._diffeq_names) and (param != 'delays'):
            raise Exception("Parameter %s must be defined as a parameter in the model" % param)
    
    if criterion is None:
        criterion = GammaFactor()
    
    data = array(data)
    if criterion.type == 'spikes':
        # Make sure that 'data' is a N*2-array
        if data.ndim == 1:
            data = concatenate((zeros((len(data), 1)), data.reshape((-1, 1))), axis=1)
        spikes = data
        traces = None
        groups = int(array(data)[:, 0].max() + 1) # number of target trains
    elif criterion.type == 'trace':
        if data.ndim == 1:
            data = data.reshape((1,-1))
        spikes = None
        traces = data
        groups = data.shape[0]
    elif criterion.type == 'both':
        # TODO
        log_warn("Not implemented yet")
        pass
    inputs = input
    if inputs.ndim==1:
        inputs = inputs.reshape((1,-1))

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')

    # default overlap when no time slicing
    if slices == 1:
        overlap = 0*ms

    # check numerical integration method
    if use_gpu and method not in ['Euler', 'RK', 'exponential_Euler']:
        raise Exception("The method can only be 'Euler', 'RK', or 'exponential_Euler' when using the GPU") 
    if method not in ['Euler', 'RK', 'exponential_Euler', 'linear', 'nonlinear']:
        raise Exception("The method can only be 'Euler', 'RK', 'exponential_Euler', 'linear', or 'nonlinear'")

    # determines whether optimization over refractoriness or not
    if type(refractory) is tuple or type(refractory) is list:
        params['refractory'] = refractory
        max_refractory = refractory[-1]
    else:
        max_refractory = None
    
    # Initialize GPU
    if use_gpu:
        set_gpu_device(0)
    
    simulator = Simulator(model, reset, threshold, 
                         inputs,
                         input_var = input_var,
                         dt = dt,
                         refractory = refractory,
                         max_refractory = max_refractory,
                         spikes = spikes,
                         traces = traces,
                         groups = groups,
                         slices = slices,
                         overlap = overlap,
                         onset = overlap,
                         neurons = neurons,
                         initial_values = initial_values,
                         unit_type = unit_type,
                         stepsize = stepsize,
                         precision = precision,
                         criterion = criterion,
                         statemonitor_var = record,
                         method = method)
    criterion_values = simulator.run(**params)
    if record is not None:
        record_values = simulator.get_statemonitor_values()
        return criterion_values, record_values
    else:
        return criterion_values




if __name__ == '__main__':
    
    from brian import loadtxt, ms, savetxt, loadtxt, Equations, NeuronGroup, run, SpikeMonitor,\
         StateMonitor, Network
    from pylab import *
    
    equations = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')
    input = loadtxt('current.txt')
    
    trace = loadtxt('trace_artificial.txt')
    spikes= loadtxt('spikes_artificial.txt')
    
    # GAMMA FACTOR
    criterion = GammaFactor(delta=2*ms)
    data = spikes
#    data[:,1] += 50*ms
    
    # LP ERROR
    criterion = LpError(p=2, varname='V')
    data = trace
    
    # SIMULATE, EVALUATE CRITERION AND RECORD TRACES ON GPU
    neurons = 5
    R = [3e9]*neurons
    tau = linspace(25*ms, 30*ms, neurons)
    criterion_values, record_values = simulate(model = equations,
                                reset = 0,
                                threshold = 1,
                                data = data,
                                input = input,
                                use_gpu = True,
                                dt = .1*ms,
                                criterion = criterion,
                                record = 'V',
                                neurons = neurons,
                                R = R,
                                tau = tau,
                                )
    print criterion_values
    
    plot(trace)
    plot(record_values.transpose())
    show()
    
    