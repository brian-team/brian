from brian import *
from brian.utils.statistics import firing_rate, get_gamma_factor
from playdoh import maximize, print_results, worker
try:
    import pycuda
    from brian.library.modelfitting.gpu_modelfitting import GPUModelFitting
    can_use_gpu = True
except ImportError:
    can_use_gpu = False

__all__ = ['modelfitting', 'print_results', 'worker', 'get_spikes', 'predict']

class ModelFitting(object):
    def __init__(self, shared_data, local_data, use_gpu):
        # Gets the key,value pairs in shared_data
        for key, val in shared_data.iteritems():
            setattr(self, key, val)
        
        # shared_data['model'] is a string
        if type(self.model) is str:
            self.model = Equations(self.model)
        
        self.total_steps = int(self.duration/self.dt)
        self.use_gpu = use_gpu
        
        self.worker_index = local_data['_worker_index']
        self.neurons = local_data['_worker_size']
        self.groups = local_data['_groups']
        
        # Time slicing
        self.input = self.input[0:self.slices*(len(self.input)/self.slices)] # makes sure that len(input) is a multiple of slices
        self.duration = len(self.input)*self.dt # duration of the input
        self.sliced_steps = len(self.input)/self.slices # timesteps per slice
        self.overlap_steps = int(self.overlap/self.dt) # timesteps during the overlap
        self.total_steps = self.sliced_steps + self.overlap_steps # total number of timesteps
        self.sliced_duration = self.overlap + self.duration/self.slices # duration of the vectorized simulation
        self.N = self.neurons*self.slices # TOTAL number of neurons in this worker
    
        self.input = hstack((zeros(self.overlap_steps), self.input)) # add zeros at the beginning because there is no overlap from the previous slice

        # Prepares data (generates I_offset, spiketimes, spiketimes_offset)
        self.prepare_data()

        self.group = NeuronGroup(self.N, model = self.model, 
                                 reset = self.reset, threshold = self.threshold,
                                 refractory = self.refractory, # NEW
                                 clock = Clock(dt=self.dt))
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value
    
        # Injects current in consecutive subgroups, where I_offset have the same value
        # on successive intervals
        k = -1
        for i in hstack((nonzero(diff(self.I_offset))[0], len(self.I_offset)-1)):
            I_offset_subgroup_value = self.I_offset[i]
            I_offset_subgroup_length = i-k
            sliced_subgroup = self.group.subgroup(I_offset_subgroup_length)
            input_sliced_values = self.input[I_offset_subgroup_value:I_offset_subgroup_value + self.total_steps]
            sliced_subgroup.set_var_by_array(self.input_var, TimedArray(input_sliced_values, clock=self.group.clock))
            k = i  
            
        if self.use_gpu:
            self.mf = GPUModelFitting(self.group, self.model, self.input, self.I_offset, 
                                      self.spiketimes, self.spiketimes_offset, zeros(self.neurons), self.delta,
                                      precision=self.precision)
        else:
            self.cc = CoincidenceCounter(self.group, self.spiketimes, self.spiketimes_offset, 
                                        onset = self.onset, delta = self.delta)

    def prepare_data(self):
        """
        Generates I_offset, spiketimes, spiketimes_offset from data,
        and also target_length and target_rates.
        
        The neurons are first grouped by time slice : there are group_size*group_count
        per group/time slice
        Within each time slice, the neurons are grouped by target train : there are
        group_size neurons per group/target train
        """
        # Generates I_offset
        self.I_offset = zeros(self.N, dtype=int)
        for slice in range(self.slices):
            self.I_offset[self.neurons*slice:self.neurons*(slice+1)] = self.sliced_steps*slice
        
        # Generates spiketimes, spiketimes_offset, target_length, target_rates
        i, t = zip(*self.data)
        i = array(i)
        t = array(t)
        alls = []
        n = 0
        pointers = []
        dt = self.dt
        
        target_length = []
        target_rates = []
        model_target = []
        group_index = 0
        
        inner_groups = self.groups.keys()
        inner_groups.sort()
        for j in inner_groups:
            neurons_in_group = self.groups[j] # number of neurons in the current group and current worker
            s = sort(t[i==j])
            target_length.extend([len(s)]*neurons_in_group)
            target_rates.extend([firing_rate(s)]*neurons_in_group)
            
            for k in xrange(self.slices):
            # first sliced group : 0...0, second_train...second_train, ...
            # second sliced group : first_train_second_slice...first_train_second_slice, second_train_second_slice...
                spikeindices = (s>=k*self.sliced_steps*dt) & (s<(k+1)*self.sliced_steps*dt) # spikes targeted by sliced neuron number k, for target j
                targeted_spikes = s[spikeindices]-k*self.sliced_steps*dt+self.overlap_steps*dt # targeted spikes in the "local clock" for sliced neuron k
                targeted_spikes = hstack((-1*second, targeted_spikes, self.sliced_duration+1*second))
                model_target.extend([k+group_index*self.slices]*neurons_in_group)
                alls.append(targeted_spikes)
                pointers.append(n)
                n += len(targeted_spikes)
            group_index += 1
        pointers = array(pointers, dtype=int)
        model_target = array(hstack(model_target), dtype=int)
        
        self.spiketimes = hstack(alls)
        self.spiketimes_offset = pointers[model_target] # [pointers[i] for i in model_target]

        # Duplicates each target_length value 'group_size' times so that target_length[i]
        # is the length of the train targeted by neuron i
        self.target_length = array(target_length, dtype=int)
        self.target_rates = array(target_rates)

    def __call__(self, param_values):
        """
        Use fitparams['_delays'] to take delays into account
        """
        if '_delays' in param_values.keys():
            delays = param_values['_delays']
            del param_values['_delays']
        else:
            delays = zeros(self.neurons)
            
        # kron spike delays
        delays = kron(delays, ones(self.slices))
        
        # Sets the parameter values in the NeuronGroup object
        self.group.reinit()
        for param, value in param_values.iteritems():
#            if param == '_delays':
#                continue
            self.group.state(param)[:] = kron(value, ones(self.slices)) # kron param_values if slicing
        
        # Reinitializes the model variables
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value
            
        if self.use_gpu:
            # Reinitializes the simulation object
            self.mf.reinit_vars(self.input, self.I_offset, self.spiketimes, self.spiketimes_offset, delays)
            # LAUNCHES the simulation on the GPU
            self.mf.launch(self.duration, self.stepsize)
            coincidence_count = self.mf.coincidence_count
            spike_count = self.mf.spike_count
        else:
            # WARNING: need to sets the group at each iteration for some reason
            self.cc.source = self.group
            # Sets the spike delay values
            self.cc.spikedelays = delays
            # Reinitializes the simulation objects
            self.group.clock.reinit()
#            self.cc.reinit()
            net = Network(self.group, self.cc)
            # LAUNCHES the simulation on the CPU
            net.run(self.duration)
            coincidence_count = self.cc.coincidences
            spike_count = self.cc.model_length
        
        coincidence_count = sum(reshape(coincidence_count, (self.slices, -1)), axis=0)
        spike_count = sum(reshape(spike_count, (self.slices, -1)), axis=0)
        
        gamma = get_gamma_factor(coincidence_count, spike_count, self.target_length, self.target_rates, self.delta)
        return gamma

def modelfitting(model = None, reset = None, threshold = None,
                 refractory = None,
                 data = None, 
                 input_var = 'I', input = None, dt = None,
                 particles = 1000, iterations = 10, pso_params = None,
                 delta = 4*ms, includedelays = True,
                 slices = 1, overlap = 0*second,
                 initial_values = None,
                 verbose = True, stepsize = 100*ms,
                 use_gpu = None, max_cpu = None, max_gpu = None,
                 precision = 'double', # set to 'float' or 'double' to specify single or double precision on the GPU
                 machines = [], named_pipe = None, port = None,
                 returninfo = False,
                 **params):
    
    # Use GPU ?
    if can_use_gpu & (use_gpu is not False):
        gpu_policy = 'prefer_gpu'
    else:
        gpu_policy = 'no_gpu'

    # Make sure that 'data' is a N*2-array
    data = array(data)
    if data.ndim == 1:
        data = concatenate((zeros((len(data), 1)), data.reshape((-1,1))), axis=1)

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')
    
    # default overlap when no time slicing
    if slices == 1:
        overlap = 0*ms

    # common values
    group_size = particles # Number of particles per target train
    group_count = int(array(data)[:,0].max()+1) # number of target trains
    N = group_size*group_count # number of neurons
    duration = len(input)*dt # duration of the input    

    # WARNING: PSO-specific
    optinfo = pso_params 
    if optinfo is None:
        optinfo = [.9, 0.1, 1.5]
    optinfo = dict(omega=optinfo[0], cl=optinfo[1], cg=optinfo[2])

    shared_data = dict(model = model,
                       threshold = threshold,
                       reset = reset,
                       refractory = refractory,
                       input_var = input_var,
                       input = input,
                       dt = dt,
                       duration = duration,
                       data = data,
                       group_size = group_size,
                       group_count = group_count,
                       delta = delta,
                       slices = slices,
                       overlap = overlap,
                       returninfo = returninfo,
                       precision = precision,
                       stepsize = stepsize,
                       initial_values = initial_values,
                       onset = 0*ms)
    
    r = maximize(   ModelFitting, 
                    params,
                    shared_data = shared_data,
                    local_data = None,
                    group_size = group_size,
                    group_count = group_count,
                    iterations = iterations,
                    optinfo = optinfo,
                    machines = machines,
                    gpu_policy = gpu_policy,
                    max_cpu = max_cpu,
                    max_gpu = max_gpu,
                    named_pipe = named_pipe,
                    port = port,
                    returninfo = returninfo,
                    verbose = verbose)
    
    # r is (results, fitinfo) or (results)
    return r

def get_spikes( model = None, reset = None, threshold = None,
                input = None, input_var = 'I', dt = None,
                **params):
    
    duration = len(input)*dt
    ngroups = len(params[params.keys()[0]])
    group = NeuronGroup(N = ngroups, model = model, reset=reset, threshold=threshold, 
                        clock=Clock(dt=dt))
    group.set_var_by_array(input_var, TimedArray(input, clock=group.clock))
    for param, values in params.iteritems():
        if (param == '_delays') | (param == 'fitness'):
            continue
        group.state(param)[:] = values
        
    M = SpikeMonitor(group)
    net = Network(group, M)
    net.run(duration)
    
    return M.spikes

def predict(model = None, reset = None, threshold = None,
            data = None, delta = 4*ms,
            input = None, input_var = 'I', dt = None,
            **params):
    
    spikes = get_spikes(model = model, reset = reset, threshold = threshold,
                        input = input, input_var = input_var, dt = dt,
                        **params)
    
    ngroups = len(params[params.keys()[0]])
    gamma = zeros(ngroups)
    for i in xrange(ngroups):
        spk = [t for j,t in spikes if j==i]
        gamma[i] = gamma_factor(spk, data, delta, normalize = True, dt = dt)
    if len(gamma) == 1:
        return gamma[0]
    else:
        return gamma
    
    