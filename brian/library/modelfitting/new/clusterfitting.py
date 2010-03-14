from brian import *
from brian.utils.particle_swarm import *
from brian.utils.statistics import get_gamma_factor, firing_rate
from brian.library.modelfitting.clustertools import *
from clustersplitting import *
from fittingparameters import *
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
        self.final_results = None
                
        # Displays the number of cores used
        if self.manager.use_gpu:
            cores =  'GPUs'
        else:
            cores = 'CPUs'
        print "Using %d %s..." % (self.numprocesses, cores)
        
        # Splits local data
        local_data_splitted = self.split_data(local_data)
        
        # Sends local data to each worker
        calls = ['prepare' for _ in xrange(self.numprocesses)]
        self.manager.process_jobs(zip(calls, local_data_splitted))

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
            local['groups'] = self.cs.groups_by_worker[i] # a dictionary (group, n)
            local['use_gpu'] = self.manager.use_gpu
            k += n
            local_data_splitted.append(local)

        return local_data_splitted

    def run(self):
        # global_states[group] is the global state for the given group
        global_states = dict([(group, None) for group in xrange(self.group_count)])
        
        # Main loop : calls iterate() for each worker 
        calls = ['iterate' for _ in xrange(self.numprocesses)]
        for iter in xrange(self.iterations):
            print "Iteration", iter
            
            # The global state is sent to each worker, it should be as light
            # as possible to avoid transmission delays
            # global_states[i] is the global state of group i
            # Here, we send the global states of all the groups inside each worker,
            # so that each worker only gets the global state of the groups inside it.
            
            # splitted_global_states[i] is a dictionary (group, global_states[group])
            # for the groups inside worker i.
            splitted_global_states = []
            for i in xrange(self.numprocesses):
                splitted_global_states.append(dict([(group, global_states[group]) for group,n in self.cs.groups_by_worker[i].iteritems()]))
            # splitted_local_states[i] is a list [(group, local_state)..]
            # with one entry per subgroup within the worker
            splitted_local_states = self.manager.process_jobs(zip(calls, splitted_global_states))
            
            # Updates the global state by combining the updated local states
            # on each worker. For example, local states may contain the global position
            # found by each worker, and the new global state is simply the best
            # position among them.
            for group in xrange(self.group_count):
                # Lists the local states of the group splitted among several workers
                local_states = [splitted_local_states[w][group] for w,n in self.cs.workers_by_group[group].iteritems()]
                global_states[group] = FittingOptimization.combine_local_states(local_states)
        
        # Terminates the optimization
        calls = ['terminate' for _ in xrange(self.numprocesses)]
        self.results = self.manager.process_jobs(zip(calls, [None for _ in xrange(self.numprocesses)]))
        self.manager.finished()
        
    def get_results(self):
        # Returns the final results : a dictionary (group, (best_param_values, best_fitness)),
        # and fit info if requested
        fp = FittingParameters(**self.shared_data['fitparams'])
        final_results = dict([(name, zeros(self.group_count)) for name in fp.param_names])
        final_results['fitness'] = zeros(self.group_count)
        for group in xrange(self.group_count):
            results_group = [self.results[w][0][group] for w,n in self.cs.workers_by_group[group].iteritems()]
            
            # results_group is a list of pairs (X, fitness) (since the group can be split among several workers)
            # here we find the best X
            X_best = None
            fitness_best = -inf
            for X, fitness in results_group:
                if fitness > fitness_best:
                    X_best = X
                    fitness_best = fitness
            param_values = fp.get_param_values(X_best)
            for name in fp.param_names:
                final_results[name][group] = param_values[name][0]
            final_results['fitness'][group] = fitness_best
        
        self.final_results = final_results
        
        # Returns the information about the simulation and optimization for each worker
        if self.shared_data['returninfo']:
            final_info = dict([])
            for w in xrange(self.numprocesses):
                final_info[w] = self.results[w][1]
            return final_results, final_info
        else:
            return final_results

    def print_results(self):
        if self.final_results is None:
            self.get_results()
        print "Results:"
        for name, values in self.final_results.iteritems():
            print name
            print values
            print
        return self.final_results

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
        self.groups = self.local_data['groups']
        self.returninfo = self.shared_data['returninfo']

        self.sim = FittingSimulation(self.local_data, self.shared_data)
        
        """
        There is one optimization object per group within each worker. The 
        FittingWorker object should take care
        of calling once the FittingSimulation object to compute the fitness
        values of every particle within the worker. Then, in each group, 
        the optimization update iteration is executed, one after the other.
        """
        # Generates the initial state matrix
        self.fp = FittingParameters(**self.shared_data['fitparams'])
        initial_param_values = self.fp.get_initial_param_values(self.local_data['neurons'])
        X0 = self.fp.get_param_matrix(initial_param_values)
        Xmin, Xmax = self.fp.set_constraints()
        
        # Initializes the FittingOptimization objects (once per group)
        self.opts = dict([])
        k = 0
        for group in self.groups.keys():
            n = self.groups[group]
            self.opts[group] = FittingOptimization(
                                       self.worker_index,
                                       X0[:,k:k+n], Xmin, Xmax,
                                       self.shared_data['optparams'],
                                       self.shared_data['returninfo'])
            k += n
        
    def iterate(self, global_states):
        """
        Optimization iteration. The global states of the groups are passed and the function returns
        the new local states. The new global state, used at the next iteration,
        is computed by the manager from the local states of every worker.
        global_states is a dictionary containing the global states for each group inside the worker.
        """
        local_states = dict([])
        
        # Compute the fitness values for all the particles inside the worker.
        X = hstack([self.opts[group].X for group in self.groups.keys()])
        param_values = self.fp.get_param_values(X)
        fitness = self.sim.sim_run(param_values)
        
        # Splits the fitness values according to groups
        k = 0
        for group, global_state in global_states.iteritems():
            n = self.groups[group] # Number of particles in the group
            # Iterates each group in series
            local_states[group] = self.opts[group].iterate(fitness[k:k+n], global_state)
            k += n
        
        # Returns a dictionary (group, local state)
        return local_states
        
    def terminate(self):
        """
        Returns fitting info at the end
        """
        results = dict([(group, self.opts[group].return_result()) for group in self.groups.keys()])
        fitinfo = dict([])
        fitinfo['sim'] = self.sim.terminate()
        fitinfo['opt'] = dict([(group, self.opts[group].terminate()) for group in self.groups.keys()])
        return results, fitinfo

if __name__ == '__main__':
    model = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')
    threshold = 1
    reset = 0
    fitparams = dict(R = [1.0e9, 1.0e10],
                     tau = [1*ms, 50*ms],
                     _delays = [-100*ms, 100*ms])
    
    input = loadtxt('current.txt')
    spikes = loadtxt('spikes.txt')
    nspikes = len(spikes)
    spikes = hstack((-1*second,spikes,2*second, -1*second,spikes+50*ms,2*second))
    
    dt = .1*ms
    group_size = 500
    group_count = 2
    iterations = 3
    delta = 4*ms
    duration = len(input)*dt
    
    spiketimes_offset = hstack((zeros(group_size, dtype=int), (nspikes+2)*ones(group_size, dtype=int)))
    target_length = len(spikes)*ones(group_count*group_size)
    target_rates = len(spikes)*Hz*ones(group_count*group_size)
    
    shared_data = dict(model=model,
                       threshold=threshold,
                       reset=reset,
                       input_var='I',
                       input=input,
                       dt=dt,
                       duration=duration,
                       spiketimes=spikes,
                       group_size=group_size,
                       group_count=group_count,
                       delta=delta,
                       returninfo=False,
                       initial_values=None,
                       onset=0*ms,
                       fitparams=fitparams,
                       optparams=[.9,1.0,1.0])
    
    local_data = dict(spiketimes_offset=spiketimes_offset,
                      target_length=target_length,
                      target_rates=target_rates)

    cluster_info = dict(gpu_policy = 'prefer_gpu',
                        max_cpu = 4,
                        max_gpu = 0,
                        machines = [],
                        named_pipe = None,
                        port = None,
                        authkey = 'brian cluster tools')
    
    fm = FittingManager(shared_data, local_data, iterations, cluster_info)
    fm.run()
    fm.print_results()
    
    