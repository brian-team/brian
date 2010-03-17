from numpy import zeros, inf, kron, ones
from clustertools import ClusterManager
from optsplit import OptSplit
from optparams import OptParams
from optworker import OptWorker
from optalg_pso import OptAlg_PSO as OptAlg # TODO: better way of changing opt alg
import time

__all__ = ['OptManager']

class OptManager:
    def __init__(self, 
                 shared_data = None, 
                 local_data = None, 
                 clusterinfo = None, 
                 optinfo = None):
        """
        Handles the run of the fitting procedure on a cluster.
        shared_data is a dictionary containing the simulation and optimization
        data shared by the workers.
        local_data contains data that is specific to each worker.
        cluster_info is a dictionary containing information about the cluster.
        """
        # Default cluster info
        if clusterinfo is None:
            clusterinfo = dict([])
        if not 'gpu_policy' in clusterinfo.keys():
            clusterinfo['gpu_policy'] = 'no_gpu'
        if not 'machines' in clusterinfo.keys():
            clusterinfo['machines'] = []
        if not 'max_cpu' in clusterinfo.keys():
            clusterinfo['max_cpu'] = None
        if not 'max_gpu' in clusterinfo.keys():
            clusterinfo['max_gpu'] = None
        if not 'named_pipe' in clusterinfo.keys():
            clusterinfo['named_pipe'] = None
        if not 'port' in clusterinfo.keys():
            clusterinfo['port'] = None
        
        # Initializes the manager object
        self.manager = ClusterManager(OptWorker,
                                     shared_data,
                                     gpu_policy = clusterinfo['gpu_policy'],
                                     own_max_cpu = clusterinfo['max_cpu'],
                                     own_max_gpu = clusterinfo['max_gpu'],
                                     machines = clusterinfo['machines'],
                                     named_pipe = clusterinfo['named_pipe'],
                                     port = clusterinfo['port'],
                                     authkey = 'distopt')
        self.numprocesses = self.manager.total_processes
        
        self.shared_data = shared_data
        self.shared_data['optinfo'] = optinfo
        self.local_data = local_data
        self.optinfo = optinfo
        
        self.final_results = None
                
        # Displays the number of cores used
        if self.shared_data['verbose']:
            if self.manager.use_gpu:
                cores =  'GPU'
            else:
                cores = 'CPU'
            if self.numprocesses > 1:
                b = 's'
            else:
                b = ''
            print "Using %d %s%s..." % (self.numprocesses, cores, b)
        
        # Splits local data
        local_data_splitted = self.split_data(local_data)
        
        # Sends local data to each worker
        calls = ['prepare' for _ in xrange(self.numprocesses)]
        self.manager.process_jobs(zip(calls, local_data_splitted))

    def split_data(self, local_data):
        """
        Splits the local data among the workers:
        Returns a list local_data_splitted.
        local_data_splitted[i] is a dictionary with the same keys as local_data,
        each value being splitted from the original value, plus the following two 
        parameters : particles and groups
            * particles is the number of particles in each worker
            * groups is a list of pairs (group, n) where n is the number of 
              particles in the subgroup 'group' for worker i
        """
        local_data_splitted = []
        
        group_size = self.shared_data['group_size']
        group_count = self.shared_data['group_count']
        self.group_count = group_count
        if group_count is None:
            group_count = 1
        # Total number of particles to split among workers
        N = group_size*group_count
        
        # Splits equally the particles among the workers
        worker_size = [N/self.numprocesses for _ in xrange(self.numprocesses)]
        worker_size[-1] = int(N-sum(worker_size[:-1]))
        
        # Keeps the groups structure within the workers
        self.cs = OptSplit(worker_size, [group_size for _ in xrange(group_count)], 
                           verbose = self.shared_data['verbose'])
        
        k = 0
        for i in xrange(self.numprocesses):
            n = worker_size[i]
            local = dict()
            if local_data is not None:
                for key,val in local_data.iteritems():
                    kronval = kron(val, ones(group_size))
                    local[key] = kronval[k:k+n]
            local['worker_size'] = n
            local['worker_index'] = i
            local['groups'] = self.cs.groups_by_worker[i] # a dictionary (group, n)
            local['use_gpu'] = self.manager.use_gpu
            k += n
            local_data_splitted.append(local)

        return local_data_splitted

    def run(self):
        # global_states[group] is the global state for the given group
        global_states = dict([(group, None) for group in xrange(self.group_count)])
        
        t0 = time.clock()
        
        # Main loop : calls iterate() for each worker 
        calls = ['iterate' for _ in xrange(self.numprocesses)]
        for iter in xrange(self.optinfo['iterations']):
            if self.shared_data['verbose']:
                print "Iteration %d/%d" % (iter+1, self.optinfo['iterations'])
            
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
                global_states[group] = OptAlg.combine_local_states(local_states)
                if self.shared_data['verbose']:
                    best_fitness = OptAlg.get_best_fitness(global_states[group])
                    if self.group_count > 1:
                        sgroup = " for group %d" % group
                    else:
                        sgroup = ""
                    print "    Current best fitness%s: %.4f" % (sgroup, best_fitness)
        
        
        if self.shared_data['verbose']:
            print "Optimization terminated in %.3f seconds." % (time.clock()-t0)
            print
        
        # Terminates the optimization
        calls = ['terminate' for _ in xrange(self.numprocesses)]
        self.results = self.manager.process_jobs(zip(calls, [None for _ in xrange(self.numprocesses)]))
        self.manager.finished()
        
    def get_results(self):
        # Returns the final results : a dictionary (group, (best_param_values, best_fitness)),
        # and fit info if requested
        fp = OptParams(**self.shared_data['optparams'])
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

