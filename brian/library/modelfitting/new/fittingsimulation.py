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
        # TODO: preparation
        print "sim initialization", self.worker_index
        
        # TODO: time slicing (create I_offset)
    
    def sim_run(self, sim_params):
        # TODO: simulation
        print "sim run", sim_params, self.worker_index
        #return sim_results
        
    def terminate(self):
        # TODO: termination
        print "sim termination", self.worker_index
        return None
