from brian import *
from brian.utils.particle_swarm import *
from brian.utils.statistics import get_gamma_factor, firing_rate
from brian.library.modelfitting.clustertools import *
from brian.library.modelfitting.cluster_splitting import *
from brian.library.modelfitting.fittingparameters import *

import sys

def combine_local_states(local__state):
    # TODO
    return None

class FittingManager:
    def __init__(self, shared_data, local_data, iterations, cluster_info):
        """
        Handles the run of the fitting procedure on a cluster.
        shared_data is a dictionary containing the simulation and optimization
        data shared by the workers.
        local_data contains data that is specific to each worker.
        cluster_info is a dictionary containing information about the cluster.
        """
        # Initializes the manager object
        self.manager = ClusterManager(FittingWorker,
                                     shared_data,
                                     gpu_policy = cluster_info['gpu_policy'],
                                     own_max_cpu = cluster_info['max_cpu'],
                                     own_max_gpu = cluster_info['max_gpu'],
                                     machines = cluster_info['machines'],
                                     named_pipe = cluster_info['named_pipe'],
                                     port = cluster_info['port'],
                                     authkey = cluster_info['authkey'])
        self.numprocesses = self.manager.total_processes
        self.iterations = iterations
        self.shared_data = shared_data
        
        # Splits local data
        local_data_splitted = self.split_data(local_data)
        
        # Sends local data to each worker
        calls = ['prepare' for i in xrange(self.numprocesses)]
        self.manager.process_jobs(zip(calls, local_data_splitted))

    def run(self):
        global_state = None
        
        # Main loop : calls iterate() for each worker 
        calls = ['iterate' for i in xrange(self.numprocesses)]
        for i in xrange(self.iterations):
            global_states = [global_state for i in xrange(self.numprocesses)] 
            local_states = self.manager.process_jobs(zip(calls, global_states))
            global_state = combine_local_states(local_states)
        
        # Gets the return information if requested
        if self.shared_data['returninfo']:
            calls = ['terminate' for i in xrange(self.numprocesses)]
            fitinfo = self.manager.process_jobs(zip(calls, [None for i in xrange(self.numprocesses)]))
            returned = global_state, terminated
        else:
            returned = global_state
        
        self.manager.finished()
        
        return returned

    def split_data(self, local_data):
        """
        Splits the following data among the workers:
            neurons
            spiketimes_offset
            target_length
            target_rates
            groups
        Returns a list local_data_splitted.
        local_data_splitted[i] is a dictionary with the same keys as local_data,
            each value being splitted from the original value, plus the following two 
            parameters : neurons and groups
                neurons is the number of neurons in each worker
                groups is a list of pairs (group, n) where n is the number of 
                    particles in the subgroup 'group' for worker i
        """
        local_data_splitted = []
        # TODO
        for i in xrange(self.numprocesses):
            local = local_data
            local['neurons'] = i
            local['groups'] = [(0,i)]
            local_data_splitted.append(local)
        return local_data_splitted

class FittingWorker():
    def __init__(self, shared_data, use_gpu):
        """
        The Fitting worker manipulates the simulation and optimization features via
        the classes FittingSimulation and FittingOptimization.
        It initializes these objects at the first process_jobs() call.
        """
        self.shared_data = shared_data
    
    def process(self, (call, job)):
        """
        Calls the correct function according to the process_jobs() call occurence.
        """
        if call == 'prepare':
            # job contains the local worker data
            self.prepare(job)
        if call == 'iterate':
            # job is 'global_state' (a dictionnary specific to the optimization
            # algorithm).
            self.iterate(job)
        if call == 'terminate':
            self.terminate()
        sys.stdout.flush()
    
    def prepare(self, local_data):
        """
        Creates the Simulation and Optimization objects with both shared data
        and local data. The shared data is passed in the constructor of the worker,
        while the local data is passed at the first process_jobs() call in the manager.
        """
        self.local_data = local_data

        self.sim = FittingSimulation(self.shared_data['model'],
                                     self.shared_data['threshold'],
                                     self.shared_data['reset'],
                                     self.shared_data['input_var'],
                                     self.shared_data['input'],
                                     self.shared_data['dt'],
                                     self.shared_data['duration'],
                                     self.shared_data['onset'],
                                     self.shared_data['stepsize'],
                                     self.shared_data['spiketimes'],
                                     self.shared_data['initial_values'],
                                     self.shared_data['delta'],
                                     self.shared_data['includedelays'],
                                     self.shared_data['precision'],
                                     self.shared_data['fitparams'], # needed for optimization (converts matrix of values 
                                                                    # to dictionary of parameter values)
                                     self.shared_data['returninfo'], # True = returns fitting info at the end
                                     self.local_data['neurons'], # number of neurons within the worker
                                     self.local_data['spiketimes_offset'],
                                     self.local_data['target_length'],
                                     self.local_data['target_rates'])
        
        self.opt = FittingOptimization(self.local_data['groups'],   # groups repartition within the worker : 
                                                                    # a list of pairs (group, n) where n is the number 
                                                                    # of particles in the subgroup 'group'
                                       self.local_data['neurons'],
                                       self.sim.sim_run,
                                       self.shared_data['pso_params'],
                                       self.shared_data['returninfo'])
        
    def iterate(self, global_state):
        """
        Optimization iteration. The global state is passed and the function returns
        the new local state. The new global state, used at the next iteration,
        is computed by the manager from the local states of every worker.
        """
        local_state = self.opt.iterate(global_state)
        return local_state
        
    def terminate(self):
        """
        Returns fitting info at the end
        """
        if self.shared_data['returninfo']:
            fitinfo['sim'] = self.sim.terminate()
            fitinfo['opt'] = self.opt.terminate()
            return fitinfo
        else:
            return false

class FittingOptimization:
    def __init__(self, groups,
                       neurons,
                       fun,
                       pso_params,
                       returninfo): 
        # TODO: preparation
        print "opt preparation"
    
    def iterate(self, global_state):
        # TODO: iteration
        # ...
        #fitness = self.fun(sim_params) # sim_params : DxN matrix
        # ...
        print "opt iteration"
        local_state = None
        
        return local_state
        
    def terminate(self):
        # TODO: termination
        print "opt termination"

class FittingSimulation:
    def __init__(self, model,
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
        # TODO: preparation
        print "sim initialization"
        print "neurons", neurons
        print
        
        # TODO: time slicing (create I_offset)
    
    def sim_run(self, sim_params):
        # TODO: simulation
        print "sim run", sim_params
        print
        #return sim_results
        
    def terminate(self):
        # TODO: termination
        print "sim termination"
        return None



if __name__ == '__main__':
    
    shared_data = dict(model=None,
                       threshold=None,
                       reset=None,
                       input_var=None,
                       input=None,
                       dt=None,
                       duration=None,
                       onset=None,
                       stepsize=None,
                       spiketimes=None,
                       initial_values=None,
                       delta=None,
                       includedelays=None,
                       precision=None,
                       fitparams=None,
                       returninfo=None,
                       pso_params=None)
    
    local_data = dict(spiketimes_offset=None,
                      target_length=None,
                      target_rates=None)

    iterations = 1
    cluster_info = dict(gpu_policy = 'prefer_gpu',
                        max_cpu = None,
                        max_gpu = None,
                        machines = [],
                        named_pipe = None,
                        port = None,
                        authkey = 'brian cluster tools')
    
    fm = FittingManager(shared_data, local_data, iterations, cluster_info)
    fm.run()
    