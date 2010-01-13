from brian import *
from brian.utils.particle_swarm import *
from brian.utils.statistics import get_gamma_factor, firing_rate
from clustertools import *
from cluster_splitting import *
from fittingparameters import *
import sys
import time
try:
    import pycuda
    from gpu_modelfitting import GPUModelFitting
    can_use_gpu = True
except ImportError:
    can_use_gpu = False

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
        
        # Preparation for the simulation code
        self.model = shared_data['model']
        self.threshold = shared_data['threshold']
        self.reset = shared_data['reset']
        self.input = shared_data['input']
        self.input_var = shared_data['input_var']
        self.dt = shared_data['dt']
        self.duration = shared_data['duration']
        self.total_steps = int(self.duration/self.dt)
        self.onset = shared_data['onset']
        self.stepsize = shared_data['stepsize']
        self.spiketimes = shared_data['spiketimes']
        self.initial_values = shared_data['initial_values']
        self.delta = shared_data['delta']
        self.includedelays = shared_data['includedelays']
        if shared_data['use_gpu'] is None or shared_data['use_gpu'] is True:
            self.use_gpu = use_gpu
        else:
            self.use_gpu = False
        params = shared_data['params']
        # Loads parameters
        self.fp = FittingParameters(includedelays = self.includedelays, **params)
        self.param_names = self.fp.param_names
        
        self.optim_prepared = False
        self.function_prepared = False
    
    def optim_prepare(self, (X, pso_params, min_values, max_values, groups)):
        """
        Initializes the PSO parameters before any iteration
        """
        self.X = X
        self.V = zeros(X.shape)
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
        self.X_gbest = [None]*len(groups)
        
        # DEBUG
        self.index_gbest = -1
        
        self.optim_prepared = True
    
    def function_prepare(self, (neurons, I_offset, spiketimes_offset, target_length, target_rates)):
        """
        Prepares the simulation code. Called once before any iteration.
        neurons is the number of neurons for the current worker.
        I_offset and spiketimes_offset allow the worker to know
        what he must compute.
        """
        self.neurons = neurons
        self.I_offset = I_offset
        self.spiketimes_offset = spiketimes_offset
        self.target_length = target_length
        self.target_rates = target_rates
        
        self.group = NeuronGroup(neurons, model = self.model, 
                                 reset = self.reset, threshold = self.threshold)
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value
    
        # INJECTS CURRENT
        # Injects current in consecutive subgroups, where I_offset have the same value
        # on successive intervals
        k = -1
        for i in hstack((nonzero(diff(I_offset))[0], len(I_offset)-1)):
            I_offset_subgroup_value = I_offset[i]
            I_offset_subgroup_length = i-k
            # DEBUG
#            print I_offset_subgroup_value, I_offset_subgroup_length
            sliced_subgroup = self.group.subgroup(I_offset_subgroup_length)
            input_sliced_values = self.input[I_offset_subgroup_value:I_offset_subgroup_value + self.total_steps]
            sliced_subgroup.set_var_by_array(self.input_var, TimedArray(input_sliced_values, clock=self.group.clock))
            k = i  
        
        self.I_offset = I_offset
        self.spiketimes_offset = spiketimes_offset
        
        if self.use_gpu:
            self.mf = GPUModelFitting(self.group, self.model, self.input, self.I_offset, 
                                      self.spiketimes, self.spiketimes_offset, zeros(neurons), self.delta)
        else:
            self.cc = CoincidenceCounterBis(self.group, self.spiketimes, self.spiketimes_offset, 
                                        onset = self.onset, delta = self.delta)
        
        self.function_prepared = True
        return
    
    def simulate(self, X):
        # Gets the parameter values contained in the matrix X, excepted spike delays values
        if self.includedelays:
            param_values = self.fp.get_param_values(X[0:-1,:], includedelays = False)
        else:
            param_values = self.fp.get_param_values(X, includedelays = False)
        # Sets the parameter values in the NeuronGroup object
        self.group.reinit()
        for param, value in param_values.iteritems():
            self.group.state(param)[:] = value
        
        # Reinitializes the model variables
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value
            
        if self.use_gpu:
            # Reinitializes the simulation object
            self.mf.reinit_vars(self.input, self.I_offset, self.spiketimes, self.spiketimes_offset, X[-1,:])
            # LAUNCHES the simulation on the GPU
            self.mf.launch(self.duration, self.stepsize)
            return self.mf.coincidence_count, self.mf.spike_count
        else:
            # WARNING: need to sets the group at each iteration for some reason
            self.cc.source = self.group
            # Sets the spike delay values
            if self.includedelays:
                self.cc.spikedelays = X[-1,:]
            # Reinitializes the simulation objects
            self.group.clock.reinit()
#            self.cc.reinit()
            net = Network(self.group, self.cc)
            # LAUNCHES the simulation on the CPU
            net.run(self.duration)
        # Computes the gamma factor
        gamma = get_gamma_factor(self.cc.coincidences, self.cc.model_length, self.target_length, self.target_rates, self.delta)
        return gamma
    
    def iterate(self, X_gbests):
        # This section is executed starting from the second iteration
        # X is not updated at the very first iteration
        if X_gbests[0] is not None:
            # X_gbests_expanded is a matrix containing the best global positions
            X_gbests_expanded = expand_items(self.groups, X_gbests)
            
            # update self.X 
#            R1 = tile(rand(1,self.N), (self.D, 1))
#            R2 = tile(rand(1,self.N), (self.D, 1))
            R1 = rand(self.D, self.N)
            R2 = rand(self.D, self.N)
            
            self.V = self.pso_params[0]*self.V + \
                self.pso_params[1]*R1*(self.X_lbest-self.X) + \
                self.pso_params[2]*R2*(X_gbests_expanded-self.X)
            self.X = self.X + self.V
        
        self.X = maximum(self.X, self.min_values)
        self.X = minimum(self.X, self.max_values)
        
        # Simulation runs
        fitness = self.simulate(self.X)
        print "    Fitness: mean %.3f, max %.3f, std %.3f" % (fitness.mean(), fitness.max(), fitness.std())
        
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
            if max_fitness > self.fitness_gbest[i]:
                index_gbest = nonzero(sub_fitness == max_fitness)[0]
                if not(isscalar(index_gbest)):
                    index_gbest = index_gbest[0]
                X_gbest = self.X[:,k+index_gbest]
                self.X_gbest[i] = X_gbest
                self.fitness_gbest[i] = max_fitness
            else:
                X_gbest = self.X_gbest[i]
                max_fitness = self.fitness_gbest[i]
            result.append((self.groups[i][0], X_gbest, max_fitness))
            k += n
        
        # DEBUG
        sys.stdout.flush()
        
        return result
        
    def process(self, X_gbests):
        # Preparation for the simulation, called before the preparation for the optimization
        if not self.function_prepared:
            # Here, X_gbests contains (neurons, I_offset, spiketimes_offset)
            self.function_prepare(X_gbests)
            return
        # Called by the optimization algorithm, before any iteration
        if not self.optim_prepared:
            # Here, X_gbests contains (X, pso_params, min_values, max_values, groups)
            self.optim_prepare(X_gbests)
            return
        # PSO iteration
        results = self.iterate(X_gbests)
        return results

def optim(X0,
          worker_size,
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
    # Prepare the manager object for optimization
    # WARNING: MUST BE CALLED *AFTER* the preparation for the simulation
    manager.process_jobs(zip(X_list,
                             pso_params_list, 
                             min_values_list, 
                             max_values_list, 
                             groups_by_worker))

    X_gbest_list = [[None]*len(groups_by_worker[i]) for i in range(num_processes)]    
    for i in range(iter):
        print "Iteration %d/%d..." % (i+1, iter)
        t1 = time.clock()
        # Each worker iterates and return its best results for each of its subgroups
        # results[i] is a list of triplets (group, best_item, best_value)
        results = manager.process_jobs(X_gbest_list)
        print "    ... took %.3f seconds." % (time.clock()-t1)
        # The results of each worker must be regrouped.
        # X_gbest_list must contain the best results for each group across workers.
        best_items = cs.combine_items(results)
        for (group, X, value) in best_items:
            print "Group %d : best value = %.3f" % (group+1, value)
        print
        X_gbest_list = cs.split_items(best_items)

    return best_items

def modelfitting(model = None, reset = None, threshold = None, data = None, 
                 input_var = 'I', input = None, dt = None,
                 verbose = True, particles = 100, slices = 1, overlap = None,
                 iterations = 10, delta = None, initial_values = None, stepsize = 100*ms,
                 use_gpu = None, max_cpu = None, max_gpu = None,
                 includedelays = True,
                 machines = [], named_pipe = None,
                 **params):
    
    # Use GPU ?
    if can_use_gpu & (use_gpu is not False):
        use_gpu = True
    else:
        use_gpu = False
    
    # WARNING: multiprocessing modelfitting currently doesn't support time slicing
    slices = 1

    # Loads parameters
    fp = FittingParameters(includedelays = includedelays, **params)

    # Make sure that data is a N*2-array
    data = array(data)
    if data.ndim == 1:
        data = concatenate((zeros((len(data), 1)), data.reshape((-1,1))), axis=1)

    if dt is None:
        raise ArgumentError
    if slices == 1:
        overlap = 0*ms

    group_size = particles # Number of particles per target train
    input = input[0:slices*(len(input)/slices)] # HACK: makes sure that len(input) is a multiple of slices
    duration = len(input)*dt # duration of the input
    sliced_steps = len(input)/slices # timesteps per slice
    overlap_steps = int(overlap/dt) # timesteps during the overlap
    total_steps = sliced_steps + overlap_steps # total number of timesteps
    sliced_duration = overlap + duration/slices # duration of the vectorized simulation
    group_count = int(array(data)[:,0].max()+1) # number of target trains
    N = group_size*group_count*slices # TOTAL number of neurons

    # The neurons are first grouped by time slice : there are group_size*group_count
    #   per group/time slice
    # Within each time slice, the neurons are grouped by target train : there are
    #   group_size neurons per group/target train

    # 1. SLICES CURRENT : returns I_offset
    input = hstack((zeros(overlap_steps), input)) # add zeros at the beginning because there is no overlap from the previous slice
    I_offset = zeros(N, dtype=int)
    for slice in range(slices):
        I_offset[group_size*group_count*slice:group_size*group_count*(slice+1)] = sliced_steps*slice

    # 2. SLICES TARGET SPIKES : returns spiketimes and spiketimes_offset
    i, t = zip(*data)
    i = array(i)
    t = array(t)
    alls = []
    n = 0
    pointers = []
    
    target_length = zeros(group_count)
    target_rates = zeros(group_count)
    
    for j in range(group_count):
        s = sort(t[i==j])
        target_length[j] = len(s)
        target_rates[j] = firing_rate(s)
        for k in range(slices):
        # first sliced group : 0...0, second_train...second_train, ...
        # second sliced group : first_train_second_slice...first_train_second_slice, second_train_second_slice...
            spikeindices = (s>=k*sliced_steps*dt) & (s<(k+1)*sliced_steps*dt) # spikes targeted by sliced neuron number k, for target j
            targeted_spikes = s[spikeindices]-k*sliced_steps*dt+overlap_steps*dt # targeted spikes in the "local clock" for sliced neuron k
            targeted_spikes = hstack((-1*second, targeted_spikes, sliced_duration+1*second))
            alls.append(targeted_spikes)
            pointers.append(n)
            n += len(targeted_spikes)
    spiketimes = hstack(alls)
    pointers = array(pointers, dtype=int)
    model_target = [] # model_target[i] is the index of the first spike targetted by neuron i
    for sl in range(slices):
        for tar in range(group_count):
            model_target.append(list((sl+tar*slices)*ones(group_size)))
    model_target = array(hstack(model_target), dtype=int)
    spiketimes_offset = pointers[model_target] # [pointers[i] for i in model_target]
    spikedelays = zeros(N)

    # Duplicates each target_length value 'group_size' times so that target_length[i]
    # is the length of the train targeted by neuron i
    target_length = kron(target_length, ones(group_size))
    target_rates = kron(target_rates, ones(group_size))


    shared_data = dict(
        neurons = N, # TOTAL number of neurons
        model = model,
        threshold = threshold,
        reset = reset,
        input = input,
        input_var = input_var,
        I_offset = I_offset,
        dt = dt,
        duration = sliced_duration,
        onset = overlap,
        spiketimes = spiketimes,
        spiketimes_offset = spiketimes_offset, 
        spikedelays = spikedelays, 
        initial_values = initial_values, 
        delta = delta, 
        stepsize = stepsize, 
        includedelays = includedelays,
        use_gpu = use_gpu,
        params = params
    )
    
    if use_gpu is False:
        gpu_policy = 'no_gpu'
    else:
        gpu_policy = 'prefer_gpu'

    manager = ClusterManager(light_worker,
                             shared_data,
                             gpu_policy=gpu_policy,
                             own_max_cpu=max_cpu,
                             own_max_gpu=max_gpu,
                             machines=machines,
                             named_pipe=named_pipe)
    num_processes = manager.total_processes
    if manager.use_gpu:
        cores =  'GPUs'
    else:
        cores = 'CPUs'
    print "Using %d %s..." % (num_processes, cores)
    
    # Size of each worker
    worker_size = [N/num_processes for _ in range(num_processes)]
    worker_size[-1] = int(N-sum(worker_size[:-1])) 
    
    I_offset_list = []
    spiketimes_offset_list = []
    
    target_length_list = []
    target_rates_list = []
    
    k = 0
    for i in range(num_processes):
        n = worker_size[i]
        I_offset_list.append(I_offset[k:k+n])
        spiketimes_offset_list.append(spiketimes_offset[k:k+n])
        target_length_list.append(target_length[k:k+n])
        target_rates_list.append(target_rates[k:k+n])
        k += n
    
    # WARNING: MUST BE CALLED FIRST, *BEFORE* the preparation for the optimization
    manager.process_jobs(zip(worker_size, 
                             I_offset_list, 
                             spiketimes_offset_list, 
                             target_length_list, 
                             target_rates_list))
    
    initial_param_values = fp.get_initial_param_values(group_size*group_count)
    X0 = fp.get_param_matrix(initial_param_values)
    min_values, max_values = fp.set_constraints(group_size*group_count)
    pso_params = [.9, .5, .9]

    best_items = optim( X0,
                        worker_size,
                        iterations, 
                        manager, 
                        num_processes,
                        group_size = group_size,
                        pso_params = pso_params, 
                        min_values = min_values, 
                        max_values = max_values)
    
    manager.finished()

    best_values = []
    D = fp.param_count
    if includedelays:
        D += 1
    X = zeros((D, group_count))
    for (i, X_group, value) in best_items:
        X[:,i] = X_group
        best_values.append(value)
        
    best_params = Parameters(**fp.get_param_values(X))
    return best_params, best_values


if __name__=='__main__':
    equations = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')
    
    input = loadtxt('current.txt')
    spikes0 = loadtxt('spikes.txt')
    
    spikes = [(0, spike) for spike in spikes0]
    spikes.extend([(1, spike+.003) for spike in spikes0])
    
    t1 = time.clock()
    best_params, best_values = modelfitting(model = equations, reset = 0, threshold = 1, 
                                 data = spikes, 
                                 input = input, dt = .1*ms,
                                 use_gpu = False, max_cpu = 4, max_gpu = None,
                                 particles = 2000, iterations = 3, delta = 1*ms,
                                 R = [1.0e9, 1.0e10], tau = [1*ms, 50*ms])
    
    print "Model fitting terminated, total duration %.3f seconds" % (time.clock()-t1)
    print
    
    print best_params
    print best_values

