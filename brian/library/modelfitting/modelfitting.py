from brian.utils.particle_swarm import *
from brian.units import check_units, second
from brian.stdunits import ms, Hz
from brian.reset import NoReset
from brian.network import Network, run
from brian.clock import reinit_default_clock
from brian.utils.parameters import Parameters
from brian.monitor import CoincidenceCounter, CoincidenceCounterBis
from brian.utils.statistics import firing_rate
from numpy import *
from numpy.random import rand, randn
from brian.neurongroup import VectorizedNeuronGroup
try:
    import pycuda
    from brian.experimental.cuda.gpu_modelfitting import GPUModelFitting
    use_gpu = True
except ImportError:
    use_gpu = False
#use_gpu = False

__all__ = ['modelfitting']

def get_param_names(params):
    """
    Returns the sorted list of parameter names.
    """
    param_names = sort(params.keys())
    return param_names

def get_initial_param_values(params, N):
    """
    Samples random initial param values around default values
    """
    random_param_values = {}
    for key, value in params.iteritems():
        if len(value) == 3:
            # One default value, value = [min, init, max]
            random_param_values[key] = value[1]*(1+.5*randn(N))
        elif len(value) == 4:
            # One default interval, value = [min, init_min, init_max, max]
            random_param_values[key] = value[1] + (value[2]-value[1])*rand(N)
        else:  
            raise ArgumentError, "Param values length should be 3 or 4"
    return random_param_values

def get_param_values(X, param_names, includedelays = False):
    """
    Converts a matrix containing param values into a dictionary
    """
    param_values = {}
    for i in range(len(param_names)):
        name = param_names[i]
        param_values[name] = X[i,:]
    if includedelays:
        # Last row in X = delays
        param_values['delays'] = X[-1,:]
    return param_values

def get_matrix(param_values, param_names):
    """
    Converts a dictionary containing param values into a matrix
    """
    X = zeros((len(param_names)+1, len(param_values[param_names[0]])))
    for i in range(len(param_names)):
        X[i,:] = param_values[param_names[i]]
    # Last row in X = delays
    X[-1,:] = param_values['delays']
    return X

def set_constraints(N = None, **params):
    """
    Returns constraints of a given model
    constraints is an array of length p where p is the number of parameters
    constraints[i] is the minimum value for parameter i
    """
    min_values = []
    max_values = []
    param_names = get_param_names(params)
    p = len(param_names)
    for key in param_names:
        value = params[key]
        min_values.append(value[0])
        max_values.append(value[-1])
        
    # Boundary conditions for delays parameter
    min_values.append(-5.0*ms)
    max_values.append(5.0*ms)
    
    min_values = array(min_values)
    max_values = array(max_values)
    min_values = tile(min_values.reshape((p+1, 1)), (1, N))
    max_values = tile(max_values.reshape((p+1, 1)), (1, N))
    return min_values, max_values

@check_units(delta=second)
def modelfitting(model = None, reset = NoReset(), threshold = None, data = None, 
                 input_var = 'I', input = None,
                 verbose = False, particles = 10, slices = 1, overlap = None,
                 iterations = 10, delta = None, init = None, stepsize = 100*ms,
                 **params):
    """
    Fits a neuron model to data.
    
    Usage example:
    params = modelfitting(model="dv/dt=-v/tau : volt", reset=0*mV, threshold=10*mV,
                               data=data, # data = [(i,t),...,(j,s)] 
                               input=I, 
                               dt=0.1*ms,
                               tau=(0*ms,5*ms,10*ms,30*ms),
                               C=(0*pF,1*pF,2*pF,inf),
                               verbose=False
                               )
    
    Parameters:
    - model         Neuron model equations
    - reset         Neuron model reset
    - threshold     Neuron model threshold
    - data          A list of spike times (i,t)
    - input         The input current (a list of values)
    - input_var     The input variable name in the equations ('I' by default) 
    - **params      Model parameters list : tau=(min,init_min,init_max,max)
    - verbose       Print iterations?
    - particles     Number of particles in the particle swarm algorithm
    - slices        Number of time slices, 1 by default
    - delta         Time window
    - init          Initial values : dictionary (state variable=initial value)
    
    Outputs:
    - params        The parameter values found by the optimization process
    - values        The gamma factor values of the best parameters found
    """
    
    param_names = get_param_names(params)
    
    if array(data).ndim == 1:
        data = concatenate((zeros((len(data), 1)), array(data).reshape((-1,1))), axis=1)
    
    NTarget = int(array(data)[:,0].max()+1)
    # N is the number of neurons
    # There are 'particles' neurons per target spike train
    N = particles * NTarget
    
    initial_param_values = get_initial_param_values(params, N)
    
    if use_gpu:
        slices = 1
        
    vgroup = VectorizedNeuronGroup(model = model, threshold = threshold, reset = reset, 
                 input_var = input_var, input = input,
                 slices = slices, overlap = overlap, init = init,
                 **initial_param_values)
    model_target = kron(arange(NTarget), ones(particles))
    
    I = array(input)
    I_offset = zeros(len(vgroup), dtype=int)
    i, t = zip(*data)
    i = array(i)
    t = array(t)
    alls = []
    n = 0
    pointers = []
    for j in xrange(amax(i)+1):
        s = sort(t[i==j])
        s = hstack((-1*second, s, vgroup.duration+1*second))
        alls.append(s)
        pointers.append(n)
        n += len(s)
    spiketimes = hstack(alls)
    pointers = array(pointers, dtype=int)
    spiketimes_offset = pointers[array(model_target, dtype=int)] # [pointers[i] for i in model_target]
    spikedelays = zeros(len(vgroup))
    
#    cd = CoincidenceCounter(vgroup, data, model_target = model_target, delta = delta)
    cd = CoincidenceCounterBis(vgroup, spiketimes, spiketimes_offset, spikedelays, delta = delta)
    
    if use_gpu:
        
        # compute firing rates of target trains
        target_length = zeros(NTarget)
        target_rates = zeros(NTarget)
        for i in range(NTarget):
            target_train = [t for j,t in data if j == i]
            target_length[i] = len(target_train)
            target_rates[i] = firing_rate(target_train)
        target_length = target_length[array(model_target, dtype=int)]
        target_rates = target_rates[array(model_target, dtype=int)]

        mf = GPUModelFitting(vgroup, model,
                             I, I_offset, spiketimes, spiketimes_offset,
                             spikedelays,
                             delta)
        def fun(X):
            
            # bug if vgroup is not redefined at each fun call
            vgroup = VectorizedNeuronGroup(model = model, threshold = threshold, reset = reset, 
                 input_var = input_var, input = input,
                 slices = slices, overlap = overlap, init = init,
                 **initial_param_values)
            
            param_values = get_param_values(X[0:-1,:], param_names)
            vgroup.set_param_values(param_values)
            spikedelays = X[-1,:]
            mf.reinit_vars(I, I_offset, spiketimes, spiketimes_offset, spikedelays)
            mf.launch(vgroup.duration, stepsize)
            cc = mf.coincidence_count
            sc = mf.spike_count
            cd.model_length = sc
            cd.coincidences = cc
            cd.target_length = target_length
            cd.target_rates = target_rates
            gamma = cd.gamma
            return gamma
    else:
        def fun(X):
            
            # bug if vgroup is not redefined at each fun call
            vgroup = VectorizedNeuronGroup(model = model, threshold = threshold, reset = reset, 
                 input_var = input_var, input = input,
                 slices = slices, overlap = overlap, init = init, values_number = N)
            cd = CoincidenceCounterBis(vgroup, spiketimes, spiketimes_offset, spikedelays, delta = delta)
            
            param_values = get_param_values(X[0:-1,:], param_names)
            vgroup.set_param_values(param_values)
            net = Network(vgroup, cd)
            reinit_default_clock()
            cd.reinit()
            cd.spikedelays = X[-1,:]
            net.run(vgroup.duration)
            gamma = cd.gamma
            return gamma
    
    
    initial_param_values['delays'] = -5*ms + 10*ms*rand(N)
    X0 = get_matrix(initial_param_values, param_names)
    min_values, max_values = set_constraints(N = N, **params)
    
    X, value, T = particle_swarm(X0, fun, iterations = iterations, pso_params = [.9, 1.9, 1.9], 
                     min_values = min_values, max_values = max_values, 
                     group_size = particles)
    
    best_params = get_param_values(X, param_names, True) 
    
    return Parameters(**best_params), value
