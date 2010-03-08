from brian import *


def combine_local_states(local_states):
    # TODO
    return None




class FittingOptimization:
    """
    TODO: there should be one optimization object per group within each
    worker to simplify things. The FittingWorker object should take care
    of calling once the FittingSimulation object to compute the fitness
    values of every neuron within the worker. Then, in each group, 
    the optimization update iteration is executed, one after the other.
    """
    def __init__(self, neurons,
                       pso_params,
                       returninfo):
        self.neurons = neurons
        self.pso_params = pso_params
        self.returninfo = returninfo
        
        # TODO: preparation (generation of the initial values)
        pass
    
    def iterate(self, fitness, global_state):
        """
        The FittingManager object handles the computation of the fitness function
        for all the groups within the worker at each iteration. fitness here
        is the vector of the fitness values for the neurons in a specific group.
        """
        # TODO: iteration
        # ...
        #fitness = self.fun(sim_params) # sim_params : DxN matrix
        # ...
        pass
        local_state = None
        
        return local_state
        
    def terminate(self):
        # TODO: termination
        pass
