
from brian import Equations, NeuronGroup, Clock, CoincidenceCounter, Network, zeros, array, \
                    ones, kron, ms, second, concatenate, hstack, sort, nonzero, diff, TimedArray, \
                    reshape, sum, log, Monitor, NetworkOperation, defaultclock, linspace, vstack, \
                    arange, sort_spikes, rint, SpikeMonitor, Connection,SpikeGeneratorGroup,DelayConnection,\
                    SpikeCounter,forget,diagflat,inf,sqrt
from scipy.sparse import dia_matrix,lil_matrix
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
import sys, cPickle

__all__ = ['modelfitting', 'print_table', 'get_spikes', 'predict', 'PSO', 'GA','CMAES','Brette','VanRossum',
           'MAXCPU', 'MAXGPU',
           'GammaFactor', 'LpError',
           'debug_level', 'info_level', 'warning_level', 'open_server']



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




class Criterion(Monitor, NetworkOperation):
    """
    Abstract class from which modelfitting criterions should derive.
    Derived classes should implement the following methods:
    
    ``initialize(self, **params)``
        Called once before the simulation. ```params`` is a dictionary with criterion-specific
        parameters. 
    
    ``timestep_call(self)``
        Called at every timestep.
    
    ``spike_call(self, neurons)``
        Called at every spike, with the spiking neurons as argument.
    
    ``get_values(self)``
        Called at the end, returns the criterion values.
        
    You have access to the following methods:
    
    ``self.get_value(self, varname)``
        Returns the value of the specified variable for all neurons (vector).
    
    You have access to the following attributes:
    
    ``self.step``
        The time step (integer)
    
    ``self.group``
        The NeuronGroup
    
    ``self.traces=None``
        Target traces.
        A 2-dimensional K*T array where K is the number of targets, and T the total number of timesteps.
        It is still a 2-dimensional array when K=1. It is None if not specified.
        
    ``self.spikes=None``
        A list of target spike trains : [(i,t)..] where i is the target index and t the spike time.
    
    ``self.N``
        The number of neurons in the NeuronGroup
    
    ``self.K=1``
        The number of targets.
    
    ``self.duration``
        The total duration of the simulation, in seconds.
    
    ``self.total_steps``
        The total number of time steps.
    
    ``self.dt``
        The timestep duration, in seconds.
    
    ``self.delays=zeros(self.n)``
        The delays for every neuron, in seconds. The delay is relative to the target.
    
    ``self.onset=0*ms``
        The onset, in seconds. The first timesteps, before onset, should be discarded in the criterion.
    
    ``self.intdelays=0``
        The delays, but in number of timesteps (``int(delays/dt)``).
    
    NOTE: traces and spikes have all been sliced before being passed to the criterion object
    """
    def __init__(self, group, traces=None, spikes=None, targets_count=1, duration=None, onset=0*ms, 
                 spikes_inline=None, spikes_offset=None,
                 traces_inline=None, traces_offset=None,
                 delays=None, when='end', **params):
        NetworkOperation.__init__(self, None, clock=group.clock, when=when)
        self.group = group
        
        # needed by SpikeMonitor
        self.source = group
        self.source.set_max_delay(0)
        self.delay = 0
        
        self.N = len(group) # number of neurons
        self.K = targets_count # number of targets
        self.dt = self.clock.dt
        if traces is not None:
            self.traces = array(traces) # KxT array
            if self.traces.ndim == 1:
                self.traces = self.traces.reshape((1,-1))
            assert targets_count==self.traces.shape[0]
        self.spikes = spikes
        # get the duration from the traces if duration is not specified in the constructor
        if duration is None: duration = self.traces.shape[1] # total number of steps
        self.duration = duration
        self.total_steps = int(duration/self.dt)
        if delays is None: delays = zeros(self.n)
        self.delays = delays
        self.onset = int(onset/self.dt)
        self.intdelays = array(self.delays/self.clock.dt, dtype=int)
        self.mindelay = min(delays)
        self.maxdelay = max(delays)
        # the following data is sliced
        self.spikes_inline = spikes_inline
        self.spikes_offset = spikes_offset
        self.traces_inline = traces_inline
        self.traces_offset = traces_offset
        if self.spikes is not None:
            # target spike count and rates
            self.target_spikes_count = self.get_spikes_count(self.spikes)
            self.target_spikes_rate = self.get_spikes_rate(self.spikes)
        self.initialize(**params)

    def step(self):
        """
        Return the current time step
        """
        return int(self.clock.t/self.dt)
    
    def get_spikes_count(self, spikes):
        count = zeros(self.K)
        for (i,t) in spikes:
            count[i] += 1
        return count
    
    def get_spikes_rate(self, spikes):
        count = self.get_spikes_count(spikes)
        return count*1.0/self.duration
    
    def initialize(self): # TO IMPLEMENT
        """
        Override this method to initialize the criterion before the simulation
        """
        pass
    
    def timestep_call(self): # TO IMPLEMENT
        """
        Override this method to do something at every time step
        """
        pass
    
    def spike_call(self, neurons): # TO IMPLEMENT
        """
        Override this method to do something at every time spike. neurons contains
        the list of neurons that just spiked
        """
        pass
    
    def get_value(self, varname):
        return self.group.state_(varname)
    
    def __call__(self):
        self.timestep_call()
        
    def propagate(self, neurons):
        self.spike_call(neurons)

    def get_values(self): # TO IMPLEMENT
        """
        Override this method to return the criterion values at the end.
        It must return one (or several, as a tuple) additive values, i.e.,
        values corresponding to slices. The method normalize can normalize
        combined values.
        """
        pass
    additive_values = property(get_values)

    def normalize(self, values): # TO IMPLEMENT
        """
        Values contains combined criterion values. It is either a vector of values (as many values
        as neurons), or a tuple with different vector of as many values as neurons.
        """
        pass

class CriterionStruct(object):
    type = None # 'trace', 'spike' or 'both'
    def get_name(self):
        return self.__class__.__name__
    name = property(get_name)




class LpErrorCriterion(Criterion):
    def initialize(self, p=2, varname='v'):
        self.p = p
        self.varname = varname
        self._error = zeros((self.K, self.N))
    
    def timestep_call(self):
        v = self.get_value(self.varname)
        t = self.step()
        if t<self.onset: return # onset
        d = self.intdelays
        indices = (t-d>=0)&(t-d<self.total_steps) # neurons with valid delays (stay inside the target trace)
        vtar = self.traces[:,t-d] # target value at this timestep for every neuron
        for i in xrange(self.K):
            self._error[i,indices] += abs(v[indices]-vtar[i,indices])**self.p
    
    def get_values(self):
        if self.K == 1: error = self._error.flatten()
        else: error = self._error
        return error # just the integral, for every slice
    
    def normalize(self, error):
        # error is now the combined error on the whole duration (sum on the slices)
        # HACK: 1- because modelfitting MAXIMIZES for now...
        return 1-self.dt*error**(1./self.p)

class LpError(CriterionStruct):
    """
    Structure used by the users to specify a criterion
    """
    def __init__(self, p = 2, varname = 'v'):
        self.type = 'trace'
        self.p = p
        self.varname = varname




class GammaFactorCriterion(Criterion):
    """
    Coincidence counter class.
    
    Counts the number of coincidences between the spikes of the neurons in the network (model spikes),
    and some user-specified data spike trains (target spikes). This number is defined as the number of 
    target spikes such that there is at least one model spike within +- ``delta``, where ``delta``
    is the half-width of the time window.
    
    Initialised as::
    
        cc = CoincidenceCounter(source, data, delta = 4*ms)
    
    with the following arguments:
    
    ``source``
        A :class:`NeuronGroup` object which neurons are being monitored.
    
    ``data``
        The list of spike times. Several spike trains can be passed in the following way.
        Define a single 1D array ``data`` which contains all the target spike times one after the
        other. Now define an array ``spiketimes_offset`` of integers so that neuron ``i`` should 
        be linked to target train: ``data[spiketimes_offset[i]], data[spiketimes_offset[i]+1]``, etc.
        
        It is essential that each spike train with the spiketimes array should begin with a spike at a
        large negative time (e.g. -1*second) and end with a spike that is a long time
        after the duration of the run (e.g. duration+1*second).
    
    ``delta=4*ms``
        The half-width of the time window for the coincidence counting algorithm.
    
    ``spiketimes_offset``
        A 1D array, ``spiketimes_offset[i]`` is the index of the first spike of 
        the target train associated to neuron i.
        
    ``spikedelays``
        A 1D array with spike delays for each neuron. All spikes from the target 
        train associated to neuron i are shifted by ``spikedelays[i]``.
        
    ``coincidence_count_algorithm``
        If set to ``'exclusive'``, the algorithm cannot count more than one
        coincidence for each model spike.
        If set to ``'inclusive'``, the algorithm can count several coincidences
        for a single model spike.
    
    ``onset``
        A scalar value in seconds giving the start of the counting: no
        coincidences are counted before ``onset``.
    
    Has three attributes:
    
    ``coincidences``
        The number of coincidences for each neuron of the :class:`NeuronGroup`.
        ``coincidences[i]`` is the number of coincidences for neuron i.
        
    ``model_length``
        The number of spikes for each neuron. ``model_length[i]`` is the spike
        count for neuron i.
        
    ``target_length``
        The number of spikes in the target spike train associated to each neuron.
    """
    def initialize(self, delta, coincidence_count_algorithm):
        self.coincidence_count_algorithm = coincidence_count_algorithm
        self.delta = int(rint(delta / self.dt))
        
        self.spike_count = zeros(self.N, dtype='int')
        self.coincidences = zeros(self.N, dtype='int')
        self.spiketime_index = self.spikes_offset
        self.last_spike_time = array(rint(self.spikes_inline[self.spiketime_index] / self.dt), dtype=int)
        self.next_spike_time = array(rint(self.spikes_inline[self.spiketime_index + 1] / self.dt), dtype=int)

        # First target spikes (needed for the computation of 
        #   the target train firing rates)
#        self.first_target_spike = zeros(self.N)

        self.last_spike_allowed = ones(self.N, dtype='bool')
        self.next_spike_allowed = ones(self.N, dtype='bool')
        
    def spike_call(self, spiking_neurons):
        dt = self.dt
        t = self.step()*dt
        
        spiking_neurons = array(spiking_neurons)
        if len(spiking_neurons):
            
            if t >= self.onset:
                self.spike_count[spiking_neurons] += 1

            T_spiking = array(rint((t + self.delays[spiking_neurons]) / dt), dtype=int)

            remaining_neurons = spiking_neurons
            remaining_T_spiking = T_spiking
            while True:
                remaining_indices, = (remaining_T_spiking > self.next_spike_time[remaining_neurons]).nonzero()
                if len(remaining_indices):
                    indices = remaining_neurons[remaining_indices]
                    self.spiketime_index[indices] += 1
                    self.last_spike_time[indices] = self.next_spike_time[indices]
                    self.next_spike_time[indices] = array(rint(self.spikes_inline[self.spiketime_index[indices] + 1] / dt), dtype=int)
                    if self.coincidence_count_algorithm == 'exclusive':
                        self.last_spike_allowed[indices] = self.next_spike_allowed[indices]
                        self.next_spike_allowed[indices] = True
                    remaining_neurons = remaining_neurons[remaining_indices]
                    remaining_T_spiking = remaining_T_spiking[remaining_indices]
                else:
                    break

            # Updates coincidences count
            near_last_spike = self.last_spike_time[spiking_neurons] + self.delta >= T_spiking
            near_next_spike = self.next_spike_time[spiking_neurons] - self.delta <= T_spiking
            last_spike_allowed = self.last_spike_allowed[spiking_neurons]
            next_spike_allowed = self.next_spike_allowed[spiking_neurons]
            I = (near_last_spike & last_spike_allowed) | (near_next_spike & next_spike_allowed)

            if t >= self.onset:
                self.coincidences[spiking_neurons[I]] += 1

            if self.coincidence_count_algorithm == 'exclusive':
                near_both_allowed = (near_last_spike & last_spike_allowed) & (near_next_spike & next_spike_allowed)
                self.last_spike_allowed[spiking_neurons] = last_spike_allowed & -near_last_spike
                self.next_spike_allowed[spiking_neurons] = (next_spike_allowed & -near_next_spike) | near_both_allowed
            
    def get_values(self):
        return (self.coincidences, self.spike_count)
    
    def normalize(self, values):
        coincidence_count = values[0]
        spike_count = values[1]
        delta = self.delta*self.dt

        gamma = get_gamma_factor(coincidence_count, spike_count, 
                                 self.target_spikes_count, self.target_spikes_rate, 
                                 delta)
        
        return gamma

class GammaFactor(CriterionStruct):
    def __init__(self, delta = 4*ms, coincidence_count_algorithm = 'exclusive'):
        self.type = 'spikes'
        self.delta = delta
        self.coincidence_count_algorithm = coincidence_count_algorithm




class VanRossumCriterion(Criterion):
    def initialize(self, tau):
        self.delay_range =max(self.delays)- min(self.delays)#delay range
        self.min_delay = abs(min(self.delays))#minimum of possible delay
        self.distance_vector=zeros(self.N) 
        self.nbr_neurons_group = self.N/self.K
        
        eqs="""
        dv/dt=(-v)/tau: volt
        """
        # network to convolve target spikes with the kernel
        self.input_target=SpikeGeneratorGroup(self.K,self.spikes,clock=self.group.clock)
        self.kernel_target=NeuronGroup(self.K,model=eqs,clock=self.group.clock)
        self.C_target = DelayConnection(self.input_target, self.kernel_target, 'v', structure='dense',  max_delay=self.min_delay)     
        self.C_target.connect_one_to_one(self.input_target,self.kernel_target)
        self.C_target.delay = self.min_delay*ones_like(self.C_target.delay)

        # network to convolve population spikes with the kernel
        self.kernel_population=NeuronGroup(self.N,model=eqs,clock=self.group.clock)
        self.C_population = DelayConnection(self.group, self.kernel_population, 'v', structure='sparse',  max_delay=self.delay_range)
        for iN in xrange(self.N):
            self.C_population.delay[iN,iN] = diagflat(self.min_delay + self.delays[iN])
        self.C_population.connect_one_to_one(self.group,self.kernel_population)
        self.spikecount = SpikeCounter(self.group)
        self.contained_objects = [self.kernel_population,self.C_population,self.spikecount,self.input_target,self.C_target,self.kernel_target]  

    def __call__(self):
        trace_population = self.kernel_population.state_('v')
        trace_target = self.kernel_target.state_('v')
        for igroup in xrange(self.K):
            self.distance_vector[igroup*self.nbr_neurons_group:(1+igroup)*self.nbr_neurons_group] += (trace_population[igroup*self.nbr_neurons_group:(1+igroup)*self.nbr_neurons_group]-trace_target[igroup])**2

    def get_values(self):
        return (self.distance_vector)
    
    def normalize(self, distance_vector):
        distance_vector[nonzero(self.spikecount.count==0)] = inf
        return -self.distance_vector*self.group.clock.dt


class VanRossum(CriterionStruct):
    def __init__(self, tau):
        self.type = 'spikes'
        self.tau = tau


class BretteCriterion(Criterion):
    def initialize(self,tau_metric):
        self.delay_range =max(self.delays)- min(self.delays)#delay range
        self.min_delay = abs(min(self.delays))#minimum of possible delay
        self.corr_vector=zeros(self.N) 
        self.norm_pop = zeros(self.N) 
        self.norm_target = zeros(self.N) 
        self.nbr_neurons_group = self.N/self.K
        
        eqs="""
        tau:second
        dv/dt=(-v)/tau: volt
        """
        # network to convolve target spikes with the kernel
        self.input_target=SpikeGeneratorGroup(self.K,self.spikes,clock=self.group.clock)
        self.kernel_target=NeuronGroup(self.N,model=eqs,clock=self.group.clock)
        self.C_target = DelayConnection(self.input_target, self.kernel_target, 'v', structure='sparse',  max_delay=self.min_delay)  
        self.kernel_target.tau=tau_metric
        for igroup in xrange(self.K):
            self.C_target.W[igroup,igroup*self.nbr_neurons_group:(1+igroup)*self.nbr_neurons_group] = ones(self.nbr_neurons_group)
            self.C_target.delay[igroup,igroup*self.nbr_neurons_group:(1+igroup)*self.nbr_neurons_group] =  self.min_delay * ones(self.nbr_neurons_group)
            
        # network to convolve population spikes with the kernel
        self.kernel_population=NeuronGroup(self.N,model=eqs,clock=self.group.clock)
        self.C_population = DelayConnection(self.group, self.kernel_population, 'v', structure='sparse',  max_delay=self.delay_range)
        for iN in xrange(self.N):
            self.C_population.delay[iN,iN] = diagflat(self.min_delay + self.delays[iN])
        self.C_population.connect_one_to_one(self.group,self.kernel_population)
        self.kernel_population.tau=tau_metric
        self.spikecount = SpikeCounter(self.group)
        self.contained_objects = [self.kernel_population,self.C_population,self.spikecount,self.input_target,self.C_target,self.kernel_target]  

    def __call__(self):
        trace_population = self.kernel_population.state_('v')
        trace_target = self.kernel_target.state_('v')
        self.corr_vector += trace_population*trace_target
        self.norm_pop += trace_population**2
        self.norm_target += trace_target**2
        
    def get_values(self):
        return (self.corr_vector,self.norm_pop,self.norm_target)
    
    def normalize(self, values):
        corr_vector=values[0]
        norm_pop=values[1]
        norm_target=values[2]
        corr_vector[nonzero(self.spikecount.count==0)] = -inf
        #print self.corr_vector/sqrt(norm_pop)/sqrt(norm_target)
        return self.corr_vector/sqrt(norm_pop)/sqrt(norm_target)


class Brette(CriterionStruct):
    def __init__(self):
        self.type = 'spikes'



class ModelFitting(Fitness):
    def initialize(self, **kwds):
        # Initialization of variables
        self.use_gpu = self.unit_type=='GPU'
        # Gets the key,value pairs in shared_data
        for key, val in self.shared_data.iteritems():
            setattr(self, key, val)
        # Gets the key,value pairs in **kwds
        for key, val in kwds.iteritems():
            setattr(self, key, val)
        self.neurons = self.nodesize
        self.groups = self.groups
        self.model = cPickle.loads(self.model)
        if type(self.model) is str:
            self.model = Equations(self.model)
        
        self.initialize_neurongroup()
        self.transform_data()
        self.inject_input()
        
#        if self.use_gpu:
#            ########
#            # TODO
#            ########
#            # Select integration scheme according to method
#            if self.method == 'Euler': scheme = euler_scheme
#            elif self.method == 'RK': scheme = rk2_scheme
#            elif self.method == 'exponential_Euler': scheme = exp_euler_scheme
#            else: raise Exception("The numerical integration method is not valid")
#            
#            self.mf = GPUModelFitting(self.group, self.model, self.input, self.I_offset,
#                                      self.spiketimes, self.spiketimes_offset, zeros(self.neurons), 0*ms, self.delta,
#                                      precision=self.precision, scheme=scheme)
#        else:
#            self.cc = CoincidenceCounter(self.group, self.spiketimes, self.spiketimes_offset,
#                                        onset=self.onset, delta=self.delta)
    
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
    
    def initialize_criterion(self, delays,tau_metric = None):
        # general criterion parameters
        params = dict(group=self.group, traces=self.sliced_traces, spikes=self.sliced_spikes, 
                      targets_count=self.groups*self.slices, duration=self.sliced_duration, onset=self.onset, 
                      spikes_inline=self.spikes_inline, spikes_offset=self.spikes_offset,
                      traces_inline=self.traces_inline, traces_offset=self.traces_offset,
                      delays=delays, when='start')
        
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
            
        if criterion_name == 'VanRossum':
            params['tau'] = self.criterion.tau
            self.criterion_object = VanRossumCriterion(**params)
            
        if criterion_name == 'Brette':
            params['tau_metric'] = tau_metric
            self.criterion_object = BretteCriterion(**params)

    
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
    
    def evaluate(self, **param_values):
        """
        Use fitparams['delays'] to take delays into account
        Use fitparams['refractory'] to take refractory into account
        """
        delays = param_values.pop('delays', zeros(self.neurons))
        refractory = param_values.pop('refractory', zeros(self.neurons))
        tau_metric = param_values.pop('tau_metric', zeros(self.neurons))

        # repeat spike delays and refractory to take slices into account
        delays = kron(delays, ones(self.slices))
        refractory = kron(refractory, ones(self.slices))
        tau_metric = kron(tau_metric, ones(self.slices))
        
        self.update_neurongroup(**param_values)
        if self.criterion.__class__.__name__ == 'Brette':
            self.initialize_criterion(delays,tau_metric)
        else:
            self.initialize_criterion(delays)
        
        if self.use_gpu:
            pass
            #########
            # TODO
            #########
#            # Reinitializes the simulation object
#            self.mf.reinit_vars(self.input, self.I_offset, self.spiketimes, self.spiketimes_offset, delays, refractory)
#            # LAUNCHES the simulation on the GPU
#            self.mf.launch(self.duration, self.stepsize)
#            coincidence_count = self.mf.coincidence_count
#            spike_count = self.mf.spike_count
        else:
            # set the refractory period
            if self.max_refractory is not None:
                self.group.refractory = refractory
            
            # Launch the simulation on the CPU
            self.group.clock.reinit()
            net = Network(self.group, self.criterion_object)
            net.run(self.duration)
        
        sliced_values = self.criterion_object.get_values()
        combined_values = self.combine_sliced_values(sliced_values)
        values = self.criterion_object.normalize(combined_values)
        return values




def modelfitting(model=None,
                 reset=None,
                 threshold=None,
                 refractory=0*ms,
                 data=None,
                 input_var='I',
                 input=None,
                 dt=None,
                 popsize=1000,
                 maxiter=10,
                 slices=1,
                 overlap=0*second,
                 initial_values=None,
                 stepsize=100 * ms,
                 unit_type='CPU',
                 total_units=None,
                 cpu=None,
                 gpu=None,
                 precision='double', # set to 'float' or 'double' to specify single or double precision on the GPU
                 machines=[],
                 allocation=None,
                 returninfo=False,
                 scaling=None,
                 algorithm=CMAES,
                 criterion=None,
                 optparams={},
                 method='Euler',
                 **params):
    """
    Model fitting function.
    
    Fits a spiking neuron model to electrophysiological data (injected current and spikes).
    
    See also the section :ref:`model-fitting-library` in the user manual.
    
    **Arguments**
    
    ``model``
        An :class:`~brian.Equations` object containing the equations defining the model.
    
    ``reset``
        A reset value for the membrane potential, or a string containing the reset
        equations.
        
    ``threshold``
        A threshold value for the membrane potential, or a string containing the threshold
        equations.
        
    ``refractory``
        The refractory period in second. If it's a single value, the same refractory will be
        used in all the simulations. If it's a list or a tuple, the fitting will also
        optimize the refractory period (see ``**params`` below).
        
        Warning: when using a refractory period, you can't use a custom reset, only a fixed one.
        
    ``data``
        A list of spike times, or a list of several spike trains as a list of pairs (index, spike time)
        if the fit must be performed in parallel over several target spike trains. In this case,
        the modelfitting function returns as many parameters sets as target spike trains.
        
    ``input_var='I'``
        The variable name used in the equations for the input current.
        
    ``input``
        A vector of values containing the time-varying signal the neuron responds to (generally
        an injected current).
        
    ``dt``
        The time step of the input (the inverse of the sampling frequency).
        
    ``**params``
        The list of parameters to fit the model with. Each parameter must be set as follows:
        ``param_name=[bound_min, min, max, bound_max]``
        where ``bound_min`` and ``bound_max`` are the boundaries, and ``min`` and ``max``
        specify the interval from which the parameter values are uniformly sampled at
        the beginning of the optimization algorithm.
        If not using boundaries, set ``param_name=[min, max]``.
        
        Also, you can add a fit parameter which is a spike delay for all spikes :
        add the special parameter ``delays`` in ``**params``, for example 
        ``modelfitting(..., delays=[-10*ms, 10*ms])``.
        
        You can also add fit the refractory period by specifying 
        ``modelfitting(..., refractory=[-10*ms, 10*ms])``.
        
    ``popsize``
        Size of the population (number of particles) per target train used by the optimization algorithm.
        
    ``maxiter``
        Number of iterations in the optimization algorithm.
        
    ``optparams``
        Optimization algorithm parameters. It is a dictionary: keys are parameter names,
        values are parameter values or lists of parameters (one value per group). 
        This argument is specific to the optimization
        algorithm used. See :class:`PSO`, :class:`GA`, :class:`CMAES`. 
        
    ``delta=4*ms``
        The precision factor delta (a scalar value in second).
        
    ``slices=1``
        The number of time slices to use.
        
    ``overlap=0*ms``
        When using several time slices, the overlap between consecutive slices, in seconds.
        
    ``initial_values``
        A dictionary containing the initial values for the state variables.
        
    ``cpu``
        The number of CPUs to use in parallel. It is set to the number of CPUs in the machine by default.
        
    ``gpu``
        The number of GPUs to use in parallel. It is set to the number of GPUs in the machine by default.
        
    ``precision``
        GPU only: a string set to either ``float`` or ``double`` to specify whether to use
        single or double precision on the GPU. If it is not specified, it will
        use the best precision available.
        
    ``returninfo=False``
        Boolean indicating whether the modelfitting function should return technical information
        about the optimization.
        
    ``scaling=None``
        Specify the scaling used for the parameters during the optimization. 
        It can be ``None`` or ``'mapminmax'``. It is ``None``
        by default (no scaling), and ``mapminmax`` by default for the CMAES algorithm.
        
     ``algorithm=CMAES``
         
         
     ``optparams={}``
         Optimization parameters. See
         
    ``method='Euler'``
        Integration scheme used on the CPU and GPU: ``'Euler'`` (default), ``RK``, 
        or ``exponential_Euler``. 
        See also :ref:`numerical-integration`.
        
    ``machines=[]``
        A list of machine names to use in parallel. See :ref:`modelfitting-clusters`.
    
    **Return values**
    
    Return an :class:`OptimizationResult` object with the following attributes:
    
    ``best_pos``
        Minimizing position found by the algorithm. For array-like fitness functions,
        it is a single vector if there is one group, or a list of vectors.
        For keyword-like fitness functions, it is a dictionary
        where keys are parameter names and values are numeric values. If there are several groups,
        it is a list of dictionaries.
    
    ``best_fit``
        The value of the fitness function for the best positions. It is a single value if 
        there is one group, or it is a list if there are several groups.
    
    ``info``
        A dictionary containing various information about the optimization.


    Also, the following syntax is possible with an ``OptimizationResult`` instance ``or``.
    The ``key`` is either an optimizing parameter name for keyword-like fitness functions,
    or a dimension index for array-like fitness functions.
    
    ``or[key]``
        it is the best ``key`` parameter found (single value), or the list
        of the best parameters ``key`` found for all groups.
    
    ``or[i]``
        where ``i`` is a group index. This object has attributes ``best_pos``, ``best_fit``,
        ``info`` but only for group ``i``.
    
    ``or[i][key]``
        where ``i`` is a group index, is the same as ``or[i].best_pos[key]``.

    For more details on the gamma factor, see
    `Jolivet et al. 2008, "A benchmark test for a quantitative assessment of simple neuron models", J. Neurosci. Methods <http://www.ncbi.nlm.nih.gov/pubmed/18160135>`__ (available in PDF
    `here <http://icwww.epfl.ch/~gerstner/PUBLICATIONS/Jolivet08.pdf>`__).
    """

    if criterion is None:
        criterion = GammaFactor()
        
    data = array(data)
    if criterion.type == 'spikes':
        # Make sure that 'data' is a N*2-array
        if data.ndim == 1:
            data = concatenate((zeros((len(data), 1)), data.reshape((-1, 1))), axis=1)
        spikes = data
        traces = None 
    elif criterion.type == 'trace':
        if data.ndim == 1:
            data = data.reshape((1,-1))
        spikes = None
        traces = data
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

    # default allocation
    if cpu is None and gpu is None:
        if CANUSEGPU: gpu = 1
        else: cpu = MAXCPU-1

    # check numerical integration method
    if (gpu>0 or unit_type == 'GPU') and method not in ['Euler', 'RK', 'exponential_Euler']:
        raise Exception("The method can only be 'Euler', 'RK', or 'exponential_Euler' when using the GPU") 
    if method not in ['Euler', 'RK', 'exponential_Euler', 'linear', 'nonlinear']:
        raise Exception("The method can only be 'Euler', 'RK', 'exponential_Euler', 'linear', or 'nonlinear'")

    if (algorithm == CMAES) & (scaling is None):
        scaling = 'mapminmax'
        
    # determines whether optimization over refractoriness or not
    if type(refractory) is tuple or type(refractory) is list:
        params['refractory'] = refractory
        max_refractory = refractory[-1]
#        refractory = 'refractory'
    else:
        max_refractory = None

    # common values
#    group_size = particles # Number of particles per target train
    groups = int(array(data)[:, 0].max() + 1) # number of target trains
#    N = group_size * group_count # number of neurons
    duration = len(input) * dt # duration of the input

    # keyword arguments for Modelfitting initialize
    kwds = dict(   model=cPickle.dumps(model),
                   threshold=threshold,
                   reset=reset,
                   refractory=refractory,
                   max_refractory=max_refractory,
                   input_var=input_var, dt=dt,
                   duration=duration,
                   criterion=criterion,
                   slices=slices,
                   overlap=overlap,
                   returninfo=returninfo,
                   precision=precision,
                   stepsize=stepsize,
                   method=method,
                   onset=overlap)

    shared_data = dict(inputs=inputs,
                       traces=traces,
                       spikes=spikes,
                       initial_values=initial_values)

    r = maximize(   ModelFitting,
                    shared_data=shared_data,
                    kwds = kwds,
                    groups=groups,
                    popsize=popsize,
                    maxiter=maxiter,
                    optparams=optparams,
                    unit_type = unit_type,
                    machines=machines,
                    allocation=allocation,
                    cpu=cpu,
                    gpu=gpu,
                    returninfo=returninfo,
                    codedependencies=[],
                    algorithm=algorithm,
                    scaling=scaling,
                    **params)

    # r is (results, fitinfo) or (results)
    return r




def get_spikes(model=None, reset=None, threshold=None,
                input=None, input_var='I', dt=None,
                **params):
    """
    Retrieves the spike times corresponding to the best parameters found by
    the modelfitting function.
    
    **Arguments**
    
    ``model``, ``reset``, ``threshold``, ``input``, ``input_var``, ``dt``
        Same parameters as for the ``modelfitting`` function.
        
    ``**params``
        The best parameters returned by the ``modelfitting`` function.
    
    **Returns**
    
    ``spiketimes``
        The spike times of the model with the given input and parameters.
    """
    duration = len(input) * dt
    ngroups = len(params[params.keys()[0]])

    group = NeuronGroup(N=ngroups, model=model, reset=reset, threshold=threshold,
                        clock=Clock(dt=dt))
    group.set_var_by_array(input_var, TimedArray(input, clock=group.clock))
    for param, values in params.iteritems():
        if (param == 'delays') | (param == 'fitness'):
            continue
        group.state(param)[:] = values

    M = SpikeMonitor(group)
    net = Network(group, M)
    net.run(duration)
    reinit_default_clock()
    return M.spikes

def predict(model=None, reset=None, threshold=None,
            data=None, delta=4 * ms,
            input=None, input_var='I', dt=None,
            **params):
    """
    Predicts the gamma factor of a fitted model with respect to the data with
    a different input current.
    
    **Arguments**
    
    ``model``, ``reset``, ``threshold``, ``input_var``, ``dt``
        Same parameters as for the ``modelfitting`` function.
        
    ``input``
        The input current, that can be different from the current used for the fitting
        procedure.
    
    ``data``
        The experimental spike times to compute the gamma factor against. They have
        been obtained with the current ``input``.
    
    ``**params``
        The best parameters returned by the ``modelfitting`` function.
    
    **Returns**
    
    ``gamma``
        The gamma factor of the model spike trains against the data.
        If there were several groups in the fitting procedure, it is a vector
        containing the gamma factor for each group.
    """
    spikes = get_spikes(model=model, reset=reset, threshold=threshold,
                        input=input, input_var=input_var, dt=dt,
                        **params)

    ngroups = len(params[params.keys()[0]])
    gamma = zeros(ngroups)
    for i in xrange(ngroups):
        spk = [t for j, t in spikes if j == i]
        gamma[i] = gamma_factor(spk, data, delta, normalize=True, dt=dt)
    if len(gamma) == 1:
        return gamma[0]
    else:
        return gamma



if __name__ == '__main__':
    
    from brian import loadtxt, ms, savetxt, loadtxt, Equations, NeuronGroup, run, SpikeMonitor,\
         StateMonitor, Network
    from pylab import *
    
    def generate_data():
        g = NeuronGroup(1, model=equations, reset=0, threshold=1)
        g.I = TimedArray(input, dt=.1*ms)
        g.tau = 25*ms
        g.R = 3e9
        SpM = SpikeMonitor(g)
        StM = StateMonitor(g, 'V', record=True)
        net = Network(g, SpM, StM)
        net.run(1*second)
        return StM.values[0], SpM.spikes
    
    equations = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')
    input = loadtxt('current.txt')
    
    # ARTIFICIAL DATA: R=3e9, tau=25*ms
#    spikes = loadtxt('spikes.txt') # real data
#    trace, spikes = generate_data()
#    savetxt('trace_artificial.txt', trace)
#    savetxt('spikes_artificial.txt', spikes)
    
    trace = loadtxt('trace_artificial.txt')
    spikes= loadtxt('spikes_artificial.txt')
    
    # GAMMA FACTOR
#    criterion = GammaFactor(delta=4*ms)
#    data = spikes
    
    # LP ERROR
#    criterion = LpError(p=2, varname='V')
#    data = trace
    
     ##Van Rossum ERROR
#    criterion = VanRossum(tau=4*ms)
#    data = spikes
    
    
    
    equations = Equations('''
    dV/dt=(R*I-V)/tau : 1
    I : 1
    R : 1
    tau_metric:second
    tau : second
    ''')
    # Brette ERROR
    criterion = Brette()
    data = spikes
#    
    results = modelfitting( model = equations,
                            reset = 0,
                            threshold = 1,
                            data = data,
                            input = input,
                            cpu = 1,
                            dt = .1*ms,
                            popsize = 1000,
                            maxiter = 5,
                            criterion = criterion,
                            R = [1.0e9, 9.0e9],
                            tau = [10*ms, 40*ms],
                            tau_metric=[1*ms,10*ms]
#                            delays = [-5*ms, 5*ms],
#                            refractory = [0*ms, 0*ms, 10*ms, 10*ms]
                            )
    
    
    print_table(results)
