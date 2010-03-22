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
    from gpu_modelfitting import GPUModelFitting, default_precision
    can_use_gpu = True
except ImportError:
    can_use_gpu = False
    default_precision = None

__all__ = ['modelfitting', 'modelfitting_worker']

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
    """
    Worker handling both simulations and optimization in a distributed fashion.
    Currently, the optimization algorithm is a simple version of the PSO algorithm.
    Each worker only needs to know the best positions reached by particles in other workers
    at each iteration.
    """
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
        self.precision = shared_data['precision']
        self.return_matrix = shared_data['return_matrix']
        if self.precision is None:
            self.precision = default_precision
        if shared_data['use_gpu'] is None or shared_data['use_gpu'] is True:
            self.use_gpu = use_gpu
        else:
            self.use_gpu = False
        # Loads parameters
        self.fp = shared_data['fp']
        self.param_names = self.fp.param_names
        
        self.optim_prepared = False
        self.function_prepared = False
    
    def optim_prepare(self, (D, pso_params, groups)):
        """
        Initializes the PSO parameters before any iteration.
        """
        
        # Randomize initial positions
        self.N = sum([n for (i,n) in groups])
        self.D = D
        
        initial_param_values = self.fp.get_initial_param_values(self.N)
        self.X = self.fp.get_param_matrix(initial_param_values)
        self.V = zeros((self.D, self.N))
        self.pso_params = pso_params
        min_values, max_values = self.fp.set_constraints()
        # Tiling min_values and max_values
        self.min_values = tile(min_values.reshape((-1,1)), (1, self.N))
        self.max_values = tile(max_values.reshape((-1,1)), (1, self.N))
        
        # self.groups is the list of the group sizes within the worker
        # each group is optimized separately
        self.groups = groups
        
        self.fitness_lbest = -inf*ones(self.N)
        self.fitness_gbest = -inf*ones(len(self.groups))
        self.X_lbest = self.X
        self.X_gbest = [None]*len(groups)
        
        self.optim_prepared = True
        
        if self.return_matrix:
            self.fitness_matrices = [[] for i in range(len(self.groups))]
    
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
            sliced_subgroup = self.group.subgroup(I_offset_subgroup_length)
            input_sliced_values = self.input[I_offset_subgroup_value:I_offset_subgroup_value + self.total_steps]
            sliced_subgroup.set_var_by_array(self.input_var, TimedArray(input_sliced_values, clock=self.group.clock))
            k = i  
        
        self.I_offset = I_offset
        self.spiketimes_offset = spiketimes_offset
        
        if self.use_gpu:
            self.mf = GPUModelFitting(self.group, self.model, self.input, self.I_offset, 
                                      self.spiketimes, self.spiketimes_offset, zeros(neurons), self.delta,
                                      precision=self.precision)
        else:
            self.cc = CoincidenceCounter(self.group, self.spiketimes, self.spiketimes_offset, 
                                        onset = self.onset, delta = self.delta)
        
        self.function_prepared = True
        return
    
    def simulate(self, X):
        """
        Simulates the network with parameters given in matrix X.
        Returns the list of the gamma factors for each set of parameters (each column of X).
        """
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
            #return self.mf.coincidence_count, self.mf.spike_count
            gamma = get_gamma_factor(self.mf.coincidence_count, self.mf.spike_count, self.target_length, self.target_rates, self.delta)
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
        """
        Iterates the PSO algorithm.
        ``X_gbests`` is the list of the best positions reached by the particles
        for each group running at least partially on the worker.
        Returns the best positions found by each group in the worker.
        """
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
        
        if self.return_matrix:
            # Record histogram at each iteration, for each group within the worker
            k = 0
            for i in range(len(self.groups)):
                n = self.groups[i][1]
                (hist, bin_edges) = histogram(fitness[k:k+n], 100, range=(0.0,1.0))
                self.fitness_matrices[i].append(list(hist))
                k += n
                
#        print "    Fitness: mean %.3f, max %.3f, std %.3f" % (fitness.mean(), fitness.max(), fitness.std())
        
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
#        sys.stdout.flush()
        
        return result
        
    def process(self, X_gbests):
        """
        The server sends jobs to the workers through this function.
        This function basically decides what job to do given the arguments
        sent by the server and given the current state of the worker. 
        """
        # Returns the fitness matrix at the end of the optimization
        # 1 col = 1 iteration, 1 line = all the fitness values at the current iteration
        if X_gbests == 'give_me_the_matrix':
            for i in range(len(self.groups)):
                self.fitness_matrices[i] = array(self.fitness_matrices[i]).transpose()
            return self.fitness_matrices
        # Preparation for the simulation, called before the preparation for the optimization
        if not self.function_prepared:
            # Here, X_gbests contains (neurons, I_offset, spiketimes_offset)
            self.function_prepare(X_gbests)
            return
        # Called by the optimization algorithm, before any iteration
        if not self.optim_prepared:
            # Here, X_gbests contains (D, pso_params, groups)
            self.optim_prepare(X_gbests)
            return
        # PSO iteration
        results = self.iterate(X_gbests)
        return results

def optim(D,
          worker_size,
          iter, 
          manager, 
          num_processes, 
          group_size = None,
          pso_params = None,
          return_matrix = None # TODO: pass this argument in shared_data instead, so that 
                               # the iterate function knows whether it has to record all 
                               # the fitness values at each iteration.
          ):
    """
    Runs the optimization procedure.
    """
    N = sum(worker_size)
    
    if  group_size is None:
        group_size = N
    
    group_count = N/group_size
    cs = ClusterSplitting(worker_size, [group_size]*group_count)
        
    D_list = [D]*num_processes
    pso_params_list = [pso_params]*num_processes
    groups_by_worker = cs.groups_by_worker
    
    # Passing PSO parameters to the workers
    # Prepare the manager object for optimization
    # WARNING: MUST BE CALLED *AFTER* the preparation for the simulation
    manager.process_jobs(zip(#X_list,
                             D_list,
                             pso_params_list, 
#                             min_values_list, 
#                             max_values_list, 
                             groups_by_worker))

    X_gbest_list = [[None]*len(groups_by_worker[i]) for i in range(num_processes)]
    
    total_time = 0.0
    
    for i in range(iter):
        print "Iteration %d/%d..." % (i+1, iter)
        t1 = time.clock()
        # Each worker iterates and return its best results for each of its subgroups
        # results[i] is a list of triplets (group, best_item, best_value)
        results = manager.process_jobs(X_gbest_list)
        time_iter = time.clock()-t1
        total_time += time_iter
        print "    ... took %.3f seconds." % time_iter
        # The results of each worker must be regrouped.
        # X_gbest_list must contain the best results for each group across workers.
        best_items = cs.combine_items(results)
        mean_value = 0
        for (group, X, value) in best_items:
#            print "Group %d : best value = %.3f" % (group+1, value)
            mean_value += value
        mean_value /= len(best_items)
        print "Mean best value = %.3f" % mean_value
        print
        X_gbest_list = cs.split_items(best_items)

    if return_matrix:
        fitness_matrix = zeros((0, iter))
        # fitness_matrices[i] is a list of matrices containing the histograms
        # for all iterations and each subgroup within worker i
        fitness_matrices = manager.process_jobs(['give_me_the_matrix']*num_processes)
        # Here we have to combine the matrices corresponding to one group
        # but that were splitted among several workers
        fitness_matrices2 = [zeros((100, iter)) for i in range(group_count)]
        for worker in range(len(worker_size)):
            k = 0
            for group, n in groups_by_worker[worker]:
                fitness_matrices2[group] += fitness_matrices[worker][k]
                k += 1
        return best_items, total_time/iter, fitness_matrices2
    else:
        return best_items, total_time/iter

def modelfitting(model = None, reset = None, threshold = None,
                 data = None, 
                 input_var = 'I', input = None, dt = None,
                 particles = 1000, iterations = 10, pso_params = None,
                 delta = 2*ms, includedelays = True,
                 slices = 1, overlap = None,
                 initial_values = None,
                 verbose = True, stepsize = 100*ms,
                 use_gpu = None, max_cpu = None, max_gpu = None,
                 precision=None, # set to 'float' or 'double' to specify single or double precision on the GPU
                 machines = [], named_pipe = None, port = None, authkey='brian cluster tools',
                 return_time = None,
                 return_matrix = None,
                 **params):
    '''
    Model fitting function.
    
    Fits a spiking neuron model to electrophysiological data (injected current and spikes).
    
    See also the section :ref:`model-fitting-library` in the user manual.
    
    **Arguments**
    
    ``model``
        An :class:`~brian.Equations` object containing the equations defining the model.
    ``reset``
        A reset value for the membrane potential, or a string containing the reset
        equations.
    ``threshold``
        A threshold value for the membrane potential, or a string containing the threshold
        equations.
    ``data``
        A list of spike times, or a list of several spike trains as a list of pairs (index, spike time)
        if the fit must be performed in parallel over several target spike trains. In this case,
        the modelfitting function returns as many parameters sets as target spike trains.
    ``input_var='I'``
        The variable name used in the equations for the input current.
    ``input``
        A vector of values containing the time-varying signal the neuron responds to (generally
        an injected current).
    ``dt``
        The time step of the input (the inverse of the sampling frequency).
    ``**params``
        The list of parameters to fit the model with. Each parameter must be set as follows:
        ``param_name=[bound_min, min, max, bound_max]``
        where ``bound_min`` and ``bound_max`` are the boundaries, and ``min`` and ``max``
        specify the interval from which the parameter values are uniformly sampled at
        the beginning of the optimization algorithm.
        If not using boundaries, set ``param_name=[min, max]``.
    ``particles``
        Number of particles per target train used by the particle swarm optimization algorithm.
    ``iterations``
        Number of iterations in the particle swarm optimization algorithm.
    ``pso_params``
        Parameters of the PSO algorithm. It is a list with three scalar values (omega, c_l, c_g).
        The parameter ``omega`` is the "inertial constant", ``c_l`` is the "local best"
        constant affecting how much the particle's personl best influences its movement, and
        ``c_g`` is the "global best" constant affecting how much the global best
        position influences each particle's movement. See the
        `wikipedia entry on PSO <http://en.wikipedia.org/wiki/Particle_swarm_optimization>`__
        for more details (note that they use ``c_1`` and ``c_2`` instead of ``c_l``
        and ``c_g``). Reasonable values are (.9, .5, 1.5), but experimentation
        with other values is a good idea.
    ``delta=2*ms``
        The precision factor delta (a scalar value in second).
    ``includedelays=True``
        A boolean indicating whether optimizing the spike delay or not.
    ``initial_values``
        A dictionary containing the initial values for the state variables.
    ``verbose=True``
        A boolean value indicating whether printing the progress of the optimization algorithm or not.
    ``use_gpu``
        A boolean value indicating whether using the GPU or not. This value is not taken into account
        if no GPU is present on the computer.
    ``max_cpu``
        The maximum number of CPUs to use in parallel. It is set to the number of CPUs in the machine by default.
    ``max_gpu``
        The maximum number of GPUs to use in parallel. It is set to the number of GPUs in the machine by default.
    ``precision``
        A string set to either ``float`` or ``double`` to specify whether to use
        single or double precision on the GPU. If it is not specified, it will
        use the best precision available.
    ``machines=[]``
        A list of machine names to use in parallel. See :ref:`modelfitting-clusters`.
    ``named_pipe``
        Set to ``True`` to use Windows named pipes for networking, or a string
        to use a particular name for the pipe. See :ref:`modelfitting-clusters`.
    ``port``
        The port number for IP networking, you only need to specify this if the
        default value of 2718 is blocked. See :ref:`modelfitting-clusters`.
    ``return_matrix``
        Set it to ``True`` to return the matrix of the fitness values for all particles and all iterations.
    
    **Return values**
    
    ``best_params``
        A :class:`~brian.Parameters` object containing the best parameter values for each target spike train
        found by the optimization algorithm. ``best_params[param_name]`` is a vector containing
        the parameter values for each target.
    ``best_values``
        A vector containing the best gamma factor values for each target.
        For more details on the gamma factor, see
        `Jolivet et al. 2008, "A benchmark test for a quantitative assessment of simple neuron models", J. Neurosci. Methods <http://www.ncbi.nlm.nih.gov/pubmed/18160135>`__ (available in PDF
        `here <http://icwww.epfl.ch/~gerstner/PUBLICATIONS/Jolivet08.pdf>`__).
    ``fitness_matrices``
        If ``return_matrix`` is set to ``True``, ``fitness_matrices[i]`` is a (N*iterations) matrix
        containing the histogram of the fitness values among particle within each group at each 
        iteration of the optimization algorithm.
    '''
    
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
        raise Exception('dt (sampling frequency of the input) must be set')
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

    # SLICES CURRENT : returns I_offset
    input = hstack((zeros(overlap_steps), input)) # add zeros at the beginning because there is no overlap from the previous slice
    I_offset = zeros(N, dtype=int)
    for slice in range(slices):
        I_offset[group_size*group_count*slice:group_size*group_count*(slice+1)] = sliced_steps*slice

    # SLICES TARGET SPIKES : returns spiketimes and spiketimes_offset
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
        params = params,
        fp = fp,
        precision = precision,
        return_matrix = return_matrix
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
                             named_pipe=named_pipe,
                             port=port,
                             authkey=authkey)
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
    
#    initial_param_values = fp.get_initial_param_values(group_size*group_count)
#    X0 = fp.get_param_matrix(initial_param_values)
#    min_values, max_values = fp.set_constraints()
    if pso_params is None:
        pso_params = [.9, .5, 1.5]
    D = fp.param_count
    if includedelays:
        D += 1

    if return_matrix:
        best_items, mean_iter_time, fitness_matrices = optim( D,
                                                            worker_size,
                                                            iterations, 
                                                            manager, 
                                                            num_processes,
                                                            group_size = group_size,
                                                            pso_params = pso_params,
                                                            return_matrix = True)
    else:
        best_items, mean_iter_time = optim( D,
                                            worker_size,
                                            iterations, 
                                            manager, 
                                            num_processes,
                                            group_size = group_size,
                                            pso_params = pso_params,
                                            return_matrix = False)
    manager.finished()

    best_values = []
    X = zeros((D, group_count))
    for (i, X_group, value) in best_items:
        X[:,i] = X_group
        best_values.append(value)
        
    best_params = Parameters(**fp.get_param_values(X))
    
    to_return = [best_params, best_values]
    if return_time:
        to_return.append(mean_iter_time)
    if return_matrix:
        to_return.append(fitness_matrices)
    return to_return

def modelfitting_worker(max_gpu=None, max_cpu=None, port=None, named_pipe=None,
                        authkey='brian cluster tools'):
    '''
    Model fitting worker script. See documentation in :ref:`model-fitting-library`.
    '''
    cluster_worker_script(light_worker,
                          max_gpu=max_gpu, max_cpu=max_cpu, port=port,
                          named_pipe=named_pipe, authkey=authkey)
