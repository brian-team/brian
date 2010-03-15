from brian import *
from brian.units import _get_best_unit 
from brian.library.modelfitting.clustertools import *
from clustersplitting import *
from fittingparameters import *
from fittingworker import *
from fittingsimulation import *
from fittingoptimization import *

__all__ = ['FittingManager']

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
            cores =  'GPU'
        else:
            cores = 'CPU'
        if self.numprocesses > 1:
            b = 's'
        else:
            b = ''
        print "Using %d %s%s..." % (self.numprocesses, cores, b)
        
        self.paramunits = dict(fitness=1.0)
        for param, value in self.shared_data['fitparams'].iteritems():
            self.paramunits[param] = get_unit(value[1])
        
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
            print "Iteration %d/%d" % (iter+1, self.iterations)
            
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
        else:
            final_info = None
        return final_results, final_info

    def print_results(self):
        if self.final_results is None:
            self.get_results()
        print
        print "RESULTS:"
        
        def print_quantity(x):
            if is_dimensionless(x):
                u = _get_best_unit(x*second)
                s = "%3.3f" % float(x/u)
                scale = int(round(log(float(u))/log(10)))
                if scale is not 0:
                    u = "e"+str(scale)
                else:
                    u = ''
            else:
                u = _get_best_unit(x)
                s = "%3.3f " % float(x/u)
            return s+str(u) 
        
        width = 16
        print ' '*width,
        for i in xrange(self.group_count):
            s = 'Group %d' % i
            spaces = ' '*(width-len(s))
            print s+spaces,
        print
        for name, values in self.final_results.iteritems():
            spaces = ' '*(width-len(name))
            print name+spaces,
            unit = self.paramunits[name]
            for value in values:
                s = print_quantity(value*unit)
                spaces = ' '*(width-len(s))
                print s+spaces,
            print
            
        return self.final_results
