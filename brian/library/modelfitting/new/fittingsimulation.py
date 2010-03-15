from brian import *
from brian.utils.statistics import get_gamma_factor, firing_rate

class FittingSimulation:
    def __init__(self, local_data, shared_data):
        # Gets the key,value pairs in shared_data
        for key, val in shared_data.iteritems():
            setattr(self, key, val)
        
        # shared_data['model'] is a string
        self.model = Equations(self.model)
        
        self.total_steps = int(self.duration/self.dt)
        self.use_gpu = local_data['use_gpu']
        
        self.worker_index = local_data['worker_index']
        self.neurons = local_data['neurons']
        self.groups = local_data['groups']
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
    
        # Slices data : returns spiketimes, spiketimes_offset,
        def slice_data(spiketimes, spiketimes_offset, target_length):
            """
            Slices the data
            """
            # TODO
            spiketimes_sliced = spiketimes
            spiketimes_offset_sliced = spiketimes_offset
            
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

        self.spiketimes, self.spiketimes_offset = slice_data(self.spiketimes, self.spiketimes_offset, self.target_length)

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

    def sim_run(self, param_values):
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
        
    def terminate(self):
        # TODO: termination
#        print "sim termination", self.worker_index
        return None
