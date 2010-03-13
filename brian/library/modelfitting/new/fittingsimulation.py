from brian import *




class FittingSimulation:
    def __init__(self, worker_index,
                       model,
                       threshold,
                       reset,
                       input_var,
                       input,
                       dt,
                       duration,
                       onset,
                       stepsize,
                       spiketimes,
                       initial_values,
                       delta,
                       includedelays,
                       precision,
                       fitparams,
                       returninfo,
                       neurons,
                       spiketimes_offset,
                       target_length,
                       target_rates):
        self.worker_index = worker_index
        self.neurons = neurons
        # TODO: preparation
        print "sim initialization", self.worker_index
        
        # TODO: time slicing (create I_offset)
    
    def sim_run(self, sim_params):
        """
        use fitparams['_delays'] to take delays into account
        """
        # TODO: simulation
        sim_results = exp(-sim_params['a']**2)
        
        return sim_results
        
    def terminate(self):
        # TODO: termination
        print "sim termination", self.worker_index
        return None
