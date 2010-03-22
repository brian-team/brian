from brian import *
from brian.utils.statistics import firing_rate, get_gamma_factor
from playdoh import maximize, print_results
try:
    import pycuda
    can_use_gpu = True
except ImportError:
    can_use_gpu = False

__all__ = ['modelfitting', 'print_results']

class ModelFitting(object):
    def __init__(self, shared_data, local_data, use_gpu):
        # Gets the key,value pairs in shared_data
        for key, val in shared_data.iteritems():
            setattr(self, key, val)
        
        # shared_data['model'] is a string
        self.model = Equations(self.model)
        
        self.total_steps = int(self.duration/self.dt)
        self.use_gpu = use_gpu
        
        self.worker_index = local_data['_worker_index']
        self.neurons = local_data['_worker_size']
        self.spiketimes_offset = local_data['spiketimes_offset']
        self.target_length = local_data['target_length']
        self.target_rates = local_data['target_rates']
        
        # Time slicing
        self.input = self.input[0:self.slices*(len(self.input)/self.slices)] # makes sure that len(input) is a multiple of slices
        self.duration = len(self.input)*self.dt # duration of the input
        self.sliced_steps = len(self.input)/self.slices # timesteps per slice
        self.overlap_steps = int(self.overlap/self.dt) # timesteps during the overlap
        self.total_steps = self.sliced_steps + self.overlap_steps # total number of timesteps
        self.sliced_duration = self.overlap + self.duration/self.slices # duration of the vectorized simulation
        self.N = self.neurons*self.slices # TOTAL number of neurons in this worker
    
        """
        The neurons are first grouped by time slice : there are group_size*group_count
        per group/time slice
        Within each time slice, the neurons are grouped by target train : there are
        group_size neurons per group/target train
        """
    
        # Slices current : returns I_offset
        self.input = hstack((zeros(self.overlap_steps), self.input)) # add zeros at the beginning because there is no overlap from the previous slice
        self.I_offset = zeros(self.N, dtype=int)
        for slice in range(self.slices):
            self.I_offset[self.neurons*slice:self.neurons*(slice+1)] = self.sliced_steps*slice
    
        self.spiketimes, self.spiketimes_offset = self.slice_data(self.spiketimes, self.spiketimes_offset, self.target_length)

        self.group = NeuronGroup(self.N, model = self.model, 
                                 reset = self.reset, threshold = self.threshold)
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

    def slice_data(self, spiketimes, spiketimes_offset, target_length):
        """
        Slices the data : returns spiketimes, spiketimes_offset,
        """
        # TODO
        spiketimes_sliced = spiketimes
        spiketimes_offset_sliced = array(spiketimes_offset, dtype=int)
        
#            i, t = zip(*data)
#            i = array(i)
#            t = array(t)
#            alls = []
#            n = 0
#            pointers = []
#            for j in range(group_count):
#                s = sort(t[i==j])
#                
#                last = first + target_length[j*group_size]+2
#                
#                target_length[j] = len(s)
#                target_rates[j] = firing_rate(s)
#                for k in range(slices):
#                # first sliced group : 0...0, second_train...second_train, ...
#                # second sliced group : first_train_second_slice...first_train_second_slice, second_train_second_slice...
#                    spikeindices = (s>=k*sliced_steps*dt) & (s<(k+1)*sliced_steps*dt) # spikes targeted by sliced neuron number k, for target j
#                    targeted_spikes = s[spikeindices]-k*sliced_steps*dt+overlap_steps*dt # targeted spikes in the "local clock" for sliced neuron k
#                    targeted_spikes = hstack((-1*second, targeted_spikes, sliced_duration+1*second))
#                    alls.append(targeted_spikes)
#                    pointers.append(n)
#                    n += len(targeted_spikes)
#                    
#            spiketimes = hstack(alls)
#            pointers = array(pointers, dtype=int)
#            model_target = [] # model_target[i] is the index of the first spike targetted by neuron i
#            for sl in range(slices):
#                for tar in range(group_count):
#                    model_target.append(list((sl+tar*slices)*ones(group_size)))
#            model_target = array(hstack(model_target), dtype=int)
#            spiketimes_offset = pointers[model_target] # [pointers[i] for i in model_target]
#            spikedelays = zeros(N)
        return spiketimes_sliced, spiketimes_offset_sliced
    
    def __call__(self, param_values):
        """
        Use fitparams['_delays'] to take delays into account
        """
        if '_delays' in param_values.keys():
            delays = param_values['_delays']
        else:
            delays = zeros(self.neurons)
        
        # TODO: kron param_values if slicing
        
        # Sets the parameter values in the NeuronGroup object
        self.group.reinit()
        for param, value in param_values.iteritems():
            if param == '_delays':
                continue
            self.group.state(param)[:] = value
        
        # Reinitializes the model variables
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value
            
        if self.use_gpu:
            # Reinitializes the simulation object
            self.mf.reinit_vars(self.input, self.I_offset, self.spiketimes, self.spiketimes_offset, delays)
            # LAUNCHES the simulation on the GPU
            self.mf.launch(self.duration, self.stepsize)
            #return self.mf.coincidence_count, self.mf.spike_count
            gamma = get_gamma_factor(self.mf.coincidence_count, self.mf.spike_count, self.target_length, self.target_rates, self.delta)
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
            # Computes the gamma factor
            gamma = get_gamma_factor(self.cc.coincidences, self.cc.model_length, self.target_length, self.target_rates, self.delta)
        
        # TODO: concatenates gamma
        
        return gamma

def modelfitting(model = None, reset = None, threshold = None,
                 data = None, 
                 input_var = 'I', input = None, dt = None,
                 particles = 1000, iterations = 10, pso_params = None,
                 delta = 2*ms, includedelays = True,
                 slices = 1, overlap = None,
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

    # TODO: no time slicing yet
    slices = 1

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

    # Prepares the data
    i, t = zip(*data)
    i = array(i)
    t = array(t)
    alls = []
    n = 0
    pointers = []
    target_length = zeros(group_count)
    target_rates = zeros(group_count)
    for j in range(group_count):
        s = sort(t[i==j])
        target_length[j] = len(s)
        target_rates[j] = firing_rate(s)
        s = hstack((-1*second, s, duration+1*second))
        alls.append(s)
        pointers.append(n)
        n += len(s)
    spiketimes = hstack(alls)
    pointers = array(pointers, dtype=int)
    model_target = array(arange(group_count), dtype=int)
#    model_target = array(kron(arange(group_count), ones(group_size)), dtype=int)
    spiketimes_offset = pointers[model_target]
    spikedelays = zeros(N)
#    target_length = kron(target_length, ones(group_size))
#    target_rates = kron(target_rates, ones(group_size))

    # WARNING: PSO-specific
    optinfo = pso_params 
    if optinfo is None:
        optinfo = [.9, 0.1, 1.5]
    optinfo = dict(omega=optinfo[0], cl=optinfo[1], cg=optinfo[2])

    # TODO: convert an Expression to the original string
#    if isinstance(model, Equations):
#        model = model.expr
    shared_data = dict(model = model, # MUST be a string
                       threshold = threshold,
                       reset = reset,
                       input_var = input_var,
                       input = input,
                       dt = dt,
                       duration = duration,
                       spiketimes = spiketimes,
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
    
    local_data = dict(spiketimes_offset = spiketimes_offset,
                      target_length = target_length,
                      target_rates = target_rates)
    
    r = maximize(   ModelFitting, 
                    params,
                    shared_data = shared_data,
                    local_data = local_data,
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

