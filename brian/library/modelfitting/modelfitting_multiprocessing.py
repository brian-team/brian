from brian.utils.particle_swarm import *
from brian.units import check_units, second
from brian.stdunits import ms, Hz
from brian.timedarray import TimedArray
from brian.network import Network, run
from brian.clock import reinit_default_clock, defaultclock
from brian.utils.parameters import Parameters
from brian.monitor import CoincidenceCounter, CoincidenceCounterBis
from brian.utils.statistics import firing_rate
from numpy import *
from numpy.random import rand, randn
from brian.neurongroup import NeuronGroup
from brian.utils.statistics import get_gamma_factor
from fittingparameters import FittingParameters
import multiprocessing
import itertools
try:
    import pycuda
    from brian.experimental.cuda.gpu_modelfitting import GPUModelFitting
    can_use_gpu = True
except ImportError:
    can_use_gpu = False

__all__ = ['modelfitting']

def multifun_cpu((process_n, neurons, I_offset, spiketimes_offset, X,
                  model, reset, threshold, initial_values,
                  spiketimes, overlap, delta, includedelays, fp,
                  total_steps, input, input_var, sliced_duration,
                  )):
    # PREPARE THE NEURONGROUP. We're all set to define the CoincidenceCounterBis object (CPU) or to
    # run the simulation on the GPU (same interface with spiketimes, spiketimes_offset, I, I_offset).
    group = NeuronGroup(neurons, model = model, reset = reset, threshold = threshold)
    if initial_values is not None:
        for param, value in initial_values.iteritems():
            group.state(param)[:] = value

    # INJECTS CURRENT
    # Injects current in consecutive subgroups, where I_offset have the same value
    # on successive intervals
    k = -1
    for i in hstack((nonzero(diff(I_offset))[0], len(I_offset)-1)):
        I_offset_subgroup_value = I_offset[i]
        I_offset_subgroup_length = i-k
        # DEBUG
#                print I_offset_subgroup_value, I_offset_subgroup_length
        sliced_subgroup = group.subgroup(I_offset_subgroup_length)
        input_sliced_values = input[I_offset_subgroup_value:I_offset_subgroup_value+total_steps]
        sliced_subgroup.set_var_by_array(input_var, TimedArray(input_sliced_values, clock=group.clock))
        k = i    
    
    cc = CoincidenceCounterBis(group, spiketimes, spiketimes_offset, onset = overlap, delta = delta)
    # Gets the parameter values contained in the matrix X, excepted spike delays values
    if includedelays:
        param_values = fp.get_param_values(X[0:-1,:], includedelays = False)
    else:
        param_values = fp.get_param_values(X, includedelays = False)
    # Sets the parameter values in the NeuronGroup object
    for param, value in param_values.iteritems():
        group.state(param)[:] = value
    # Sets the spike delay values
    if includedelays:
        cc.spikedelays = X[-1,:]
    # Reinitializes the simulation objects
    reinit_default_clock()
    cc.reinit()
    net = Network(group, cc)
    # LAUNCHES the simulation on the CPU
    net.run(sliced_duration)

    return cc.coincidences, cc.model_length

@check_units(delta=second)
def modelfitting(model = None, reset = None, threshold = None, data = None,
                 input_var = 'I', input = None, dt = None,
                 verbose = True, particles = 100, slices = 1, overlap = None,
                 iterations = 10, delta = None, initial_values = None, stepsize = 100*ms,
                 use_gpu = None, includedelays = True,
                 **params):
    """
    Fits a neuron model to data.

    Usage example:
    params, value = modelfitting(model="dv/dt=-v/tau : volt", reset=0*mV, threshold=10*mV,
                               data=data, # data = [(i,t),...,(j,s)]
                               input=I,
                               dt=0.1*ms,
                               tau=(0*ms,5*ms,10*ms,30*ms),
                               C=(0*pF,1*pF,2*pF,inf)
                               )

    Parameters:
    - model             Neuron model equations
    - reset             Neuron model reset
    - threshold         Neuron model threshold
    - data              A list of spike times (i,t)
    - input             The input current (a list of values)
    - input_var         The input variable name in the equations ('I' by default)
    - **params          Model parameters list : tau=(min,init_min,init_max,max)
    - verbose           Print iterations?
    - particles         Number of particles per target train in the particle swarm algorithm
    - slices            Number of time slices, 1 by default
    - delta             Time window in second
    - initial_values    Initial values : dictionary (state variable=initial value)
    - use_gpu           None : use the GPU if possible, False : do not use the GPU

    Outputs:
    - params           The parameter values found by the optimization process
    - values           The gamma factor values of the best parameters found
    """

    # Use GPU ?
    if can_use_gpu & (use_gpu is not False):
        use_gpu = True
    else:
        use_gpu = False
    if use_gpu:
        slices = 1

    # Loads parameters
    fp = FittingParameters(includedelays = includedelays, **params)
    param_names = fp.param_names

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
    sliced_duration = overlap+duration/slices # duration of the vectorized simulation
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
    for j in range(group_count):
        s = sort(t[i==j])
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

    # 3. PREPARE THE NEURONGROUP. We're all set to define the CoincidenceCounterBis object (CPU) or to
    # run the simulation on the GPU (same interface with spiketimes, spiketimes_offset, I, I_offset).
    group = NeuronGroup(N, model = model, reset = reset, threshold = threshold)
    if initial_values is not None:
        for param, value in initial_values.iteritems():
            group.state(param)[:] = value

#    for k in range(slices):
#        sliced_subgroup = group.subgroup(group_size*group_count)
#        input_sliced_values = input[I_offset[group_size*group_count*k]:I_offset[group_size*group_count*k]+total_steps]
#        sliced_subgroup.set_var_by_array(input_var, TimedArray(input_sliced_values, clock=defaultclock))

    # Computes the firing rates of target trains (needed for the computation of the gamma factor)
    target_rates = array([firing_rate(sort(t[i==j])) for j in range(group_count)])
    target_length = array([len(t[i==j]) for j in range(group_count)])
    # Duplicates each target_length value 'group_size' times so that target_length[i]
    # is the length of the train targeted by neuron i
    target_length = kron(target_length, ones(group_size))
    target_rates = kron(target_rates, ones(group_size))

    if use_gpu:
        mf = GPUModelFitting(group, model, input, I_offset, spiketimes, spiketimes_offset, spikedelays, delta)
        def fun(X):
            # Gets the parameter values contained in the matrix X, excepted spike delays values
            if includedelays:
                param_values = fp.get_param_values(X[0:-1,:], includedelays = False)
            else:
                param_values = fp.get_param_values(X, includedelays = False)
            # Sets the parameter values in the NeuronGroup object
            group.reinit()
            for param, value in param_values.iteritems():
                group.state(param)[:] = tile(value, slices)
            # Gets the spike delay values
            if includedelays:
                spikedelays = X[-1,:]
            # Reinitializes the simulation object
            mf.reinit_vars(input, I_offset, spiketimes, spiketimes_offset, spikedelays)
            # LAUNCHES the simulation on the GPU
            mf.launch(sliced_duration, stepsize)
            # Count the final number of coincidences and of model spikes
            # by summing the numbers over all time slices
            coincidences = mf.coincidence_count.reshape((slices,-1)).sum(axis=0)
            model_length = mf.spike_count.reshape((slices,-1)).sum(axis=0)
            # Computes the gamma factor
            gamma = get_gamma_factor(coincidences, model_length, target_length, target_rates, delta)
            return gamma
    else:

        def fun(X):
            numprocesses = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=numprocesses)

            # X NEEDS TO BE SLICED
            X = tile(X, (1, slices))

            # Number of neurons for each process.
            N_list = N/numprocesses*ones(numprocesses, dtype=int)
            N_list[0] += N-(N/numprocesses)*numprocesses
            N_list = list(N_list)

            # Prepares the parameters passed to each process
            I_offset_list = []
            spiketimes_offset_list = []
            X_list = []
            k = 0
            for i in range(numprocesses):
                n = N_list[i]
                I_offset_list.append(I_offset[k:k+n])
                spiketimes_offset_list.append(spiketimes_offset[k:k+n])
                X_list.append(X[:,k:k+n])
                k += n

            args = zip(range(numprocesses),
                       N_list, I_offset_list, spiketimes_offset_list, X_list,
                       itertools.repeat(model), itertools.repeat(reset),
                       itertools.repeat(threshold), itertools.repeat(initial_values),
                       itertools.repeat(spiketimes), itertools.repeat(overlap),
                       itertools.repeat(delta), itertools.repeat(includedelays),
                       itertools.repeat(fp), itertools.repeat(total_steps),
                       itertools.repeat(input), itertools.repeat(input_var),
                       itertools.repeat(sliced_duration),
                       )
            results = pool.map(multifun_cpu, args) # launches multiple processes
            
            # Concatenates the number of coincidences and model spikes computed
            # on each core.
            coincidences = array([])
            model_length = array([])
            for (local_coincidences, local_model_length) in results:
                coincidences = hstack((coincidences, local_coincidences))
                model_length = hstack((model_length, local_model_length))

            print coincidences
            print model_length

            # Count the final number of coincidences and of model spikes
            # by summing the numbers over all time slices
            coincidences = coincidences.reshape((slices,-1)).sum(axis=0)
            model_length = model_length.reshape((slices,-1)).sum(axis=0)
            # Computes the gamma factor
            gamma = get_gamma_factor(coincidences, model_length, target_length, target_rates, delta)
            return gamma

    initial_param_values = fp.get_initial_param_values(group_size*group_count)
    X0 = fp.get_param_matrix(initial_param_values)
    min_values, max_values = fp.set_constraints(group_size*group_count)

    X, value, T = particle_swarm(X0, fun, iterations = iterations, pso_params = [.9, 1.9, 1.9],
                     min_values = min_values, max_values = max_values,
                     group_size = group_size, verbose = verbose)

    best_params = fp.get_param_values(X)

    return Parameters(**best_params), value
