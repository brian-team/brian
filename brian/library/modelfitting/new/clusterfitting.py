from brian import *
from brian.utils.particle_swarm import *
from brian.utils.statistics import get_gamma_factor, firing_rate
from brian.library.modelfitting.clustertools import *
from brian.library.modelfitting.cluster_splitting import *
from brian.library.modelfitting.fittingparameters import *

class VPObject:
    """
    Allows to call an attribute with obj.attr instead of obj.data['attr']
    """
    def __init__(self, attributes):
        self._attributes = attributes
    
    def add_attributes(self, attributes):
        self._attributes.update(attributes)
    
    def __getattr__(self, item):
        try:
            return self._attributes[item]
        except KeyError:
            raise AttributeError(item)

class FittingManager:
    def __init__(self, fitting_info, optim_info, sim_info):
        #workers.init(base_info, optim_info, sim_info)
        
        '''
        SHARED DATA, PERMANENT
        
        fitting_info = 
            returninfo          # True = returns information about simulation and 
                                # optimization at the end
            param_count         # number of parameters to fit
        
        optim_info = 
            pso_params
        
        sim_info = 
            model
            threshold
            reset
            input
            input_var
            dt
            duration
            onset
            stepsize
            spiketimes
            initial_values
            delta
            includedelays
            precision
            fittingparameters   # needed for optimization (convert matrix of values 
                                # to dictionary of parameter values)
        
        
        
        NOT SHARED, PERMANENT
        
        optim_info = 
            groups              # groups repartition within the worker : 
                                  a list of pairs (group, n) where n is the number 
                                  of particles in the subgroup 'group'
        
        sim_info =
            neurons             # number of neurons within the worker
            I_offset            # may be constructed by the worker (time slicing only within workers?)
            spiketimes_offset
            target_length
            target_rates
        
        
        
        NOT SHARED, NOT PERMANENT
        
        sim_info =
            param_values        # dictionary with parameter values to use for the current optim iteration
        
        '''


        
        
        
        
        
        
        
        
        # Minimal information needed by the worker to run the simulation and optimization
        shared_data = (fitting_info, optim_info, sim_info)
        
        manager = ClusterManager(FittingWorker,
                                 shared_data,
                                 gpu_policy=gpu_policy,
                                 own_max_cpu=max_cpu,
                                 own_max_gpu=max_gpu,
                                 machines=machines,
                                 named_pipe=named_pipe,
                                 port=port,
                                 authkey=authkey)
        num_processes = manager.total_processes
        
    def run(self):
        # TODO: number of iterations
        while True:
            local_states = workers.iterate(global_state) # global_state = None at first
            global_state = combine_local_states(local_states)
        workers.terminate()
        return global_state, terminated

class FittingWorker(VPObject):
    def __init__(self, shared_data, use_gpu):
        self.prepared = False
        pass
    
    def process(self, job):
        if not(self.prepared):
            # job is a triplet (fitting_info, optim_info, sim_info)
            self.prepare(job)
        else:
            # job is 'global_state' (a dictionnary specific to the optimization
            # algorithm).
            self.iterate(job)
    
    def prepare(self, (fitting_info, optim_info, sim_info)):
        # TODO: process fitting_info
        self.sim = FittingSimulation(sim_info)
        self.opt = FittingOptimization(optim_info)
        self.sim.prepare()
        self.opt.prepare()
        self.prepared = True
        
    def iterate(self, global_state):
        local_state = self.opt.iterate(global_state)
        return local_state
        
    def terminate(self):
        pass

class FittingOptimization:
    def __init__(self, optim_info): # optim_info['fun'] = sim.run()
        pass
    
    def optim_prepare(self):
        pass
    
    def iterate(self, global_state):
        pass
        fitness = self.fun(sim_params) # sim_params : DxN matrix
        pass
        return local_state
        
    def terminate(self):
        pass

class FittingSimulation:
    def __init__(self, sim_info):
        pass
    
    def sim_prepare(self):
        pass
    
    def sim_run(self, sim_params):
        return sim_results
        
    def terminate(self):
        pass

