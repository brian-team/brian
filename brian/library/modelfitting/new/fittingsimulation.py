from brian import *
from brian.utils.statistics import get_gamma_factor, firing_rate

class FittingSimulation:
    def __init__(self, local_data, shared_data):
        # Gets the key,value pairs in shared_data
        for key, val in shared_data.iteritems():
            setattr(self, key, val)
        
        self.total_steps = int(self.duration/self.dt)
        self.use_gpu = local_data['use_gpu']
        
        self.worker_index = local_data['worker_index']
        self.neurons = local_data['neurons']
        self.groups = local_data['groups']
        self.spiketimes_offset = local_data['spiketimes_offset']
        self.target_length = local_data['target_length']
        self.target_rates = local_data['target_rates']
        
        self.group = NeuronGroup(self.neurons, model = self.model, 
                                 reset = self.reset, threshold = self.threshold)
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value
    
        # Injects current in consecutive subgroups, where I_offset have the same value
        # on successive intervals
        
        # TODO: time slicing (create I_offset)
        self.I_offset = zeros(self.neurons)
        
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
        use fitparams['_delays'] to take delays into account
        """
        if '_delays' in param_values.keys():
            delays = param_values['_delays']
        else:
            delays = zeros(self.neurons)
        
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
        
        return gamma
        
    def terminate(self):
        # TODO: termination
#        print "sim termination", self.worker_index
        return None
