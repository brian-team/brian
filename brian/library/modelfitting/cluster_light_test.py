from brian import *
from clustertools import *
from cluster_splitting import *
import sys

#def expand(N_list, group_size):
#    """
#    There are N=group_size*group_count particles, regrouped into groups of size group_size.
#    The particles must be spread among the workers, but the group structure should be kept.
#    Therefore, expand returns a list of integers lists : result[i] is the list of subgroups
#    for worker i. One group can be spread over several workers.
#    """
#    N = sum(N_list)
#    group_list = []
#    current_worker_size = 0
#    current_group_size = 0
#    previous_group_size = 0
#    groups = []
#    current_group = 0
#    current_worker = 0
#    for k in range(N):
#        current_group_size += 1
#        current_worker_size += 1
#        if current_group_size >= group_size:
#            if current_group_size - previous_group_size>0:
#                groups.append(current_group_size - previous_group_size)
#            previous_group_size = 0
#            current_group_size = 0
#            current_group += 1
#        if current_worker_size >= N_list[current_worker]:
#            if current_group_size - previous_group_size>0:
#                groups.append(current_group_size - previous_group_size)
#            group_list.append(groups)
#            current_worker_size = 0
#            previous_group_size = current_group_size
#            current_worker += 1
#            current_group = 0
#            groups = []
#    return group_list
#
#def decompress(X_list, n_list):
#    """
#    X is a list [(col vector1, number1), (col vector2, number2), ...]
#    decompress(X) is a matrix [col vector1 (number1 times), ...]
#    """
#    result = None
#    for vector, n in zip(X_list, n_list):
#        tiled = tile(vector.reshape((-1,1)),(1,n))
#        if result is None:
#            result = tiled
#        else:
#            result = hstack((result, tiled))
#    return result


def expand_items(groups, items):
    """
    If items are column vectors, expanded_items are matrices
    tiled n times along the x-axis, where n is the size of the 
    subgroup.
    groups is groups_by_worker[worker] : a list of pairs (group, n) where n
    is the number of particles in the subgroup
    """
    result = None
    # subgroups and their capacity in current worker
    groups, n_list = zip(*groups)
    
    for n, item in zip(n_list, items):
        tiled = tile(item.reshape((-1,1)),(1,n))
        if result is None:
            result = tiled
        else:
            result = hstack((result, tiled))
    return result

class light_worker(object):
    def __init__(self, shared_data, use_gpu):
        self.groups = None
        self.optim_prepared = False
    
    def optim_prepare(self, (X, fun, pso_params, min_values, max_values, groups)):
        """
        Initializes the PSO parameters before any iteration
        """
        self.X = X
        self.V = zeros(X.shape)
        self.fun = fun
        (self.D, self.N) = X.shape
        self.pso_params = pso_params
        self.min_values = min_values
        self.max_values = max_values
        # self.groups is the list of the group sizes within the worker
        # each group is optimized separately
        self.groups = groups
        
        self.fitness_lbest = -inf*ones(self.N)
        self.fitness_gbest = -inf*ones(len(groups))
        self.X_lbest = X
        
        self.optim_prepared = True
    
    def process(self, X_gbests):
        # Called by the optimization algorithm, before any iteration
        if not self.optim_prepared:
            self.optim_prepare(X_gbests)
            return
        # This section is executed starting from the second iteration
        # X is not updated at the very first iteration
        if X_gbests[0] is not None:
            # X_gbests_expanded is a matrix containing the best global positions
            X_gbests_expanded = expand_items(self.groups, X_gbests)
            
            # update self.X 
            R1 = tile(rand(1,self.N), (self.D, 1))
            R2 = tile(rand(1,self.N), (self.D, 1))
            self.V = self.pso_params[0]*self.V + \
                self.pso_params[1]*R1*(self.X_lbest-self.X) + \
                self.pso_params[2]*R2*(X_gbests_expanded-self.X)
            self.X = self.X + self.V
        
        self.X = maximum(self.X, self.min_values)
        self.X = minimum(self.X, self.max_values)
        
        fitness = self.fun(self.X)
        
        # Local update
        indices_lbest = nonzero(fitness > self.fitness_lbest)[0]
        if (len(indices_lbest)>0):
            self.X_lbest[:,indices_lbest] = self.X[:,indices_lbest]
            self.fitness_lbest[indices_lbest] = fitness[indices_lbest]
        
        # Global update
        result = []
        k = 0
        for i in range(len(self.groups)):
            group, n = self.groups[i]
            sub_fitness = fitness[k:k+n]
            max_fitness = sub_fitness.max()
            index_gbest = nonzero(sub_fitness == max_fitness)[0]
            if not(isscalar(index_gbest)):
                index_gbest = index_gbest[0]
            X_gbest = self.X[:,k+index_gbest]
            result.append((self.groups[i][0], X_gbest, max_fitness))
            k += n
        return result

def optim(X0,
          fun_list,
          iter, 
          manager, 
          num_processes, 
          group_size = None,
          pso_params = None, 
          min_values = None, 
          max_values = None):
    
    (D, N) = X0.shape
    if (min_values is None):
        min_values = -inf*ones(D)
    if (max_values is None):
        max_values = inf*ones(D)
    if  group_size is None:
        group_size = N
    
    # Number of neurons per worker
    worker_size = [N/num_processes for _ in range(num_processes)]
    worker_size[-1] = int(N-sum(worker_size[:-1])) 
    
    group_count = N/group_size
    cs = ClusterSplitting(worker_size, [group_size]*group_count)
    
    # Initial particle positions for each worker
    X_list = []
    k = 0
    for i in range(num_processes):
        n = worker_size[i]
        X_list.append(X0[:,k:k+n])
        k += n
        
    pso_params_list = [pso_params]*num_processes
    min_values_list = [tile(min_values.reshape((-1, 1)), (1, n)) for n in worker_size]
    max_values_list = [tile(max_values.reshape((-1, 1)), (1, n)) for n in worker_size]
    groups_by_worker = cs.groups_by_worker
    
    # Passing PSO parameters to the workers
    manager.process_jobs(zip(X_list,
                             fun_list,#todo
                             pso_params_list, 
                             min_values_list, 
                             max_values_list, 
                             groups_by_worker))

    X_gbest_list = [[None]*len(groups_by_worker[i]) for i in range(num_processes)]    
    for i in range(iter):
        # Each worker iterates and return its best results for each of its subgroups
        # results[i] is a list of triplets (group, best_item, best_value)
        results = manager.process_jobs(X_gbest_list)
        
        # The results of each worker must be regrouped.
        # X_gbest_list and fitness_gbest_list must contain the best results for each group across workers.
        best_items = cs.combine_items(results)
        X_gbest_list = cs.split_items(best_items)

    return best_items

def fitness_fun1(X):
    X = X**2
    return exp(-(X.sum(axis=0)))
def fitness_fun2(X):
    X = X-2
    X = X**2
    return exp(-(X.sum(axis=0)))
       
if __name__=='__main__':    
    iter = 10
    pso_params = [.9, 1.9, 1.9]
    min_values = None
    max_values = None
    group_size = 100
    group_count = 2
    total_particles = group_size*group_count
    D = 2 # number of parameters
    X0 = 3*rand(D, total_particles)
    
    fun_list = (fitness_fun1, fitness_fun2)
    
    shared_data = dict([])
    manager = ClusterManager(light_worker, shared_data, gpu_policy='no_gpu', own_max_cpu = 2)
    num_processes = manager.total_processes
    
    best_items =  optim(X0,
                        fun_list,
                        iter, 
                        manager, 
                        num_processes,
                        group_size = group_size,
                        pso_params = pso_params, 
                        min_values = min_values, 
                        max_values = max_values)
    
    manager.finished()
    
    print best_items
    