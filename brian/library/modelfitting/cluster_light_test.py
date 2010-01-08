from brian import *
from clustertools import *
import sys

def expand(N_list, group_size):
    """
    There are N=group_size*group_count particles, regrouped into groups of size group_size.
    The particles must be spread among the workers, but the group structure should be kept.
    Therefore, expand returns a list of integers lists : result[i] is the list of subgroups
    for worker i. One group can be spread over several workers.
    """
    N = sum(N_list)
    group_list = []
    current_worker_size = 0
    current_group_size = 0
    previous_group_size = 0
    groups = []
    current_group = 0
    current_worker = 0
    for k in range(N):
        current_group_size += 1
        current_worker_size += 1
        if current_group_size >= group_size:
            if current_group_size - previous_group_size>0:
                groups.append(current_group_size - previous_group_size)
            previous_group_size = 0
            current_group_size = 0
            current_group += 1
        if current_worker_size >= N_list[current_worker]:
            if current_group_size - previous_group_size>0:
                groups.append(current_group_size - previous_group_size)
            group_list.append(groups)
            current_worker_size = 0
            previous_group_size = current_group_size
            current_worker += 1
            current_group = 0
            groups = []
    return group_list

def decompress(X_list, n_list):
    """
    X is a list [(col vector1, number1), (col vector2, number2), ...]
    decompress(X) is a matrix [col vector1 (number1 times), ...]
    """
    result = None
    for vector, n in zip(X_list, n_list):
        tiled = tile(vector.reshape((-1,1)),(1,n))
        if result is None:
            result = tiled
        else:
            result = hstack((result, tiled))
    return result

class light_worker(object):
    def __init__(self, shared_data, use_gpu):
        self.X_lbest = shared_data['X']
        self.fun = shared_data['fun']
        self.groups = None
        
        self.optim_prepared = False
    
    def optim_prepare(self, (X, pso_params, min_values, max_values, groups)):
        """
        Initializes the PSO parameters before any iteration
        """
        self.X = X
        (self.D, self.N) = X.shape
        self.pso_params = pso_params
        self.min_values = min_values
        self.max_values = max_values
        # self.groups is the list of the group sizes within the worker
        # each group is optimized separately
        self.groups = groups
        self.optim_prepared = True
    
    def process(self, X_gbests):
        # Called by the optimization algorithm, before any iteration
        if not self.optim_prepared:
            self.prepare(X_gbests)
            return
        # This section is executed starting from the second iteration
        # X is not updated at the very first iteration
        if X_gbests is not None:
            
            #TODO: vectorize this to allow optimization over independent groups of particles
            
            # X_gbests is now a matrix containing the best global positions
            X_gbests = decompress(X_gbests, self.groups)
            # update self.X 
            R1 = tile(rand(1,self.N), (self.D, 1))
            R2 = tile(rand(1,self.N), (self.D, 1))
            V = self.pso_params[0]*V + \
                self.pso_params[1]*R1*(self.X_lbest-self.X) + \
                self.pso_params[2]*R2*(X_gbests-self.X)
            self.X = self.X + V
        
            self.X = maximum(self.X, self.min_values)
            self.X = minimum(self.X, self.max_values)
        
        fitness = self.fun(self.X)
        best = argmax(fitness)
        X_gbest = self.X[:,best]
        fitness_gbest = fitness[best]
        return (X_gbest, fitness_gbest)

def optim(N, 
          X0,
          iter, 
          manager, 
          num_processes, 
          group_size = None,
          pso_params = None, 
          min_values = None, 
          max_values = None):
    
    if (min_values is None):
        min_values = -inf*ones(self.D)
    if (max_values is None):
        max_values = inf*ones(self.D)
    if  group_size is None:
        group_size = 1
        
    min_values = tile(min_values.reshape((-1, 1)), (1, self.N))
    max_values = tile(max_values.reshape((-1, 1)), (1, self.N))
    
    # Number of neurons per worker
    N_list = [N/num_processes for _ in range(num_processes)]
    N_list[-1] = int(N-sum(N_list[:-1])) 
    
    # Initial particle positions for each worker
    X_list = []
    k = 0
    for i in range(num_processes):
        n = N_list[i]
        X_list.append(X0[:,k:k+n])
        k += n
        
    pso_params_list = [pso_params for i in range(num_processes)]
    min_values_list = [min_values for i in range(num_processes)]
    max_values_list = [max_values for i in range(num_processes)]
    groups_list = expand(N_list, group_size)
    
    # Passing PSO parameters to the workers
    manager.process_jobs(zip(X_list,
                             pso_params_list, 
                             min_values_list, 
                             max_values_list, 
                             groups_list))
    
    X_gbest_list = [[None for group in groups_list[i]] for i in range(num_processes)]    
    for i in range(iter):
        # Each worker iterates and return its best results for each of its subgroups
        results = manager.process_jobs(X_gbest_list)
        (X_spread, fitness_spread) = zip(*results)
        
        # The results of each worker must be regrouped.
        # fitness_list and X_list must contain the best results for each group across workers.
        fitness_list = []
        fitness = -inf
        X_list = []
        X = None
        current_group_size = 0
        for i in range(len(groups_list)):
            group = groups_list[i]
            if isscalar(group):
                group = [group]
            for j in range(len(group)):
                subgroup = group[j]
                current_group_size += subgroup
                if fitness_spread[i][j] > fitness:
                    fitness = fitness_spread[i][j]
                    X = X_spread[i][j]
                if current_group_size >= group_size:
                    fitness_list.append(fitness)
                    X_list.append(X)
                    current_group_size = 0
                    fitness = -inf
        
        # Now, the best results must be spread again across workers
        X_gbest_list = []
        X_worker = []
        current_group_size = 0
        current_group = 0
        for i in range(len(groups_list)):
            group = groups_list[i]
            if isscalar(group):
                group = [group]
            for j in range(len(group)):
                subgroup = group[j]
                X_worker.append(X_list[current_group])
                current_group_size += subgroup
                if current_group_size >= group_size:
                    current_group += 1
                    current_group_size = 0
            X_gbest_list.append(X_worker)
        
    return (X_gbest, fitness_gbest)

def fitness_fun(X):
    return exp(-X.sum(axis=0)**2)
       
if __name__=='__main__':
    
    iter = 3
    pso_params = [.9, 1.9, 1.9]
    min_values = None
    max_values = None
    total_particles = 4
    D = 2 # number of parameters
    X0 = rand((D, total_particles))
    
    shared_data = dict(fun = fitness_fun)
    manager = ClusterManager(light_worker, shared_data, own_max_cpu = 2)
    num_processes = manager.total_processes
    
    print optim(total_particles,
                X0,
                iter, 
                manager, 
                num_processes,
                group_size = None,
                pso_params = pso_params, 
                min_values = min_values, 
                max_values = max_values)
    
    manager.finished()
    