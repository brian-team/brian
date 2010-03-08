from brian import *
from brian.utils.particle_swarm import *
from brian.utils.statistics import get_gamma_factor, firing_rate
from brian.library.modelfitting.clustertools import *
from brian.library.modelfitting.cluster_splitting import *
from brian.library.modelfitting.fittingparameters import *
from fittingsimulation import FittingSimulation
from fittingoptimization import FittingOptimization
import sys

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
        self.group_count = shared_data['group_count']
        
        # Displays the number of cores used
        if self.manager.use_gpu:
            cores =  'GPUs'
        else:
            cores = 'CPUs'
        print "Using %d %s..." % (self.numprocesses, cores)
        
        # Splits local data
        local_data_splitted = self.split_data(local_data)
        
        # Sends local data to each worker
        calls = ['prepare' for i in xrange(self.numprocesses)]
        self.manager.process_jobs(zip(calls, local_data_splitted))

    def run(self):
        global_states = [None for _ in xrange(self.group_count)]
        
        # Main loop : calls iterate() for each worker 
        calls = ['iterate' for i in xrange(self.numprocesses)]
        for iter in xrange(self.iterations):
            # The global state is sent to each worker, it should be as light
            # as possible to avoid transmission delays
            # global_states[i] is the global state of group i
            # Here, we send the global states of all the groups inside each worker,
            # so that each worker only gets the global state of the groups inside it.
            for i in xrange(self.numprocesses):
                splitted_global_states[i] = [global_states[j] for j in cs.groups_by_worker[i]]
            # splitted_local_states[i] is a list [(group, local_state)..]
            # with one entry per subgroup within the worker
            splitted_local_states = self.manager.process_jobs(zip(calls, splitted_global_states))
            
            # Updates the global state by combining the updated local states
            # on each worker. For example, local states may contain the global position
            # found by each worker, and the new global state is simply the best
            # position among them.
            for i in xrange(self.group_count):
                # Lists the local states of the group splitted among several workers
                local_states = [splitted_local_states[w][i] for w in self.cs.workers_by_group[i]]
                global_states[i] = combine_local_states(local_states)
        
        # Gets the return information if requested
        if self.shared_data['returninfo']:
            calls = ['terminate' for i in xrange(self.numprocesses)]
            fitinfo = self.manager.process_jobs(zip(calls, [None for i in xrange(self.numprocesses)]))
            returned = global_state, fitinfo
        else:
            returned = global_state
        
        self.manager.finished()
        
        return returned

    def split_data(self, local_data):
        """
        Splits the following data among the workers:
            spiketimes_offset         # vector to be splitted
            target_length             # vector to be splitted
            target_rates              # vector to be splitted
            neurons                   # number of neurons per worker
            groups                    # groups per worker
        Returns a list local_data_splitted.
        local_data_splitted[i] is a dictionary with the same keys as local_data,
        each value being splitted from the original value, plus the following two 
        parameters : neurons and groups
            * neurons is the number of neurons in each worker
            * groups is a list of pairs (group, n) where n is the number of 
              particles in the subgroup 'group' for worker i
        """
        local_data_splitted = []
        
        group_size = self.shared_data['group_size']
        group_count = self.shared_data['group_count']
        if group_count is None:
            group_count = 1
        # Total number of neurons to split among workers
        N = group_size*group_count
        
        # Splits equally the neurons among the workers
        worker_size = [N/self.numprocesses for _ in xrange(self.numprocesses)]
        worker_size[-1] = int(N-sum(worker_size[:-1]))
        
        # Keeps the groups structure within the workers
        self.cs = ClusterSplitting(worker_size, [group_size for _ in xrange(group_count)])
        
        k = 0
        for i in xrange(self.numprocesses):
            n = worker_size[i]
            local = dict()
            local['spiketimes_offset'] = local_data['spiketimes_offset'][k:k+n]
            local['target_length'] = local_data['target_length'][k:k+n]
            local['target_rates'] = local_data['target_rates'][k:k+n]
            local['neurons'] = n
            local['worker_index'] = i
            local['groups'] = self.cs.groups_by_worker[i]
            k += n
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
            result = self.prepare(job)
        if call == 'iterate':
            # job is 'global_state' (a dictionnary specific to the optimization
            # algorithm).
            result = self.iterate(job)
        if call == 'terminate':
            result = self.terminate()
        sys.stdout.flush()
        return result
    
    def prepare(self, local_data):
        """
        Creates the Simulation and Optimization objects with both shared data
        and local data. The shared data is passed in the constructor of the worker,
        while the local data is passed at the first process_jobs() call in the manager.
        """
        self.local_data = local_data
        self.worker_index = local_data['worker_index']

        self.sim = FittingSimulation(self.worker_index,
                                     self.shared_data['model'],
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
        
        self.opt = FittingOptimization(self.worker_index,
                                       self.local_data['groups'],   # groups repartition within the worker : 
                                                                    # a list of pairs (group, n) where n is the number 
                                                                    # of particles in the subgroup 'group'
                                       self.local_data['neurons'],
                                       self.sim.sim_run,
                                       self.shared_data['pso_params'],
                                       self.shared_data['returninfo'])
        self.ngroups = len(self.local_data['groups'])
        
    def iterate(self, global_state):
        """
        Optimization iteration. The global state is passed and the function returns
        the new local state. The new global state, used at the next iteration,
        is computed by the manager from the local states of every worker.
        global_state is the list of the global states for each group inside the worker.
        """
        # fitness[i] is the vector of the fitness values for subgroup i inside the worker
        # TODO HERE
        local_state = [self.opt.iterate(fitness[i], global_state[i]) for i in xrange(self.ngroups)]
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

if __name__ == '__main__':
    n = 16
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
                       group_size=n,
                       group_count=2,
                       initial_values=None,
                       delta=None,
                       includedelays=None,
                       precision=None,
                       fitparams=None,
                       returninfo=None,
                       pso_params=None)
    
    local_data = dict(spiketimes_offset=ones(n),
                      target_length=ones(n),
                      target_rates=ones(n))

    iterations = 1
    cluster_info = dict(gpu_policy = 'prefer_gpu',
                        max_cpu = 4,
                        max_gpu = 0,
                        machines = [],
                        named_pipe = None,
                        port = None,
                        authkey = 'brian cluster tools')
    
    fm = FittingManager(shared_data, local_data, iterations, cluster_info)
    fm.run()
    