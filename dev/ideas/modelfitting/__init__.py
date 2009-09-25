from brian import *
from coincidence_counter import *
from optimization import *

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

def get_param_values(X, param_names):
    """
    Converts a matrix containing param values into a dictionary
    """
    param_values = {}
    for i in range(len(param_names)):
        name = param_names[i]
        param_values[name] = X[i,:]
    return param_values

def get_matrix(param_values, param_names):
    """
    Converts a dictionary containing param values into a matrix
    """
    X = zeros((len(param_names), len(param_values[param_names[0]])))
    for i in range(len(param_names)):
        X[i,:] = param_values[param_names[i]]
    return X


def fit(fun, X0, group_size, iterations = 10, min_values = None, max_values = None):
    """
    Maximizes a function starting from initial values X0
    if y=fun(x), 
        x is a D*(group_size*Ntarget) matrix
        y is a group_size*Ntarget vector
    D is the number of parameters
    group_size is the number of particles per target train
    Ntarget is the number of target trains
    fit maximizes fun independently over Ntarget subgroups
    fit returns a D*Ntarget matrix.
    """
    X, val, T = optimize(X0, fun, iterations = iterations, pso_params = [.9, 2.0, 2.0], 
                         min_values = min_values, max_values = max_values, 
                         group_size = group_size)
    return X, val


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
    min_values = array(min_values)
    max_values = array(max_values)
    min_values = tile(min_values.reshape((p, 1)), (1, N))
    max_values = tile(max_values.reshape((p, 1)), (1, N))
    return min_values, max_values


def modelfitting(model = None, reset = NoReset(), threshold = None, data = None, 
                 input_name = None, input_values = None, dt = 0.1*ms,
                 timeslices = 1, verbose = False, particle_number = 10, slice_number = 1,
                 iterations = 10, delta = None,
                 **params):
    """
    Fits a neuron model to data.
    
    Usage example:
    params, value = modelfitting(model="dv/dt=-v/tau : volt", reset=0*mV, threshold=10*mV,
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
    - dt            Timestep of the input
    - **params      Model parameters list : tau=(min,init_min,init_max,max)
    - verbose       Print iterations?
    - particle_number
                    Number of particles in the particle swarm algorithm
    - slice_number  Number of time slices, 1 by default
    - delta         Time window
    
    Outputs:
    - params        The parameter values found by the optimization process
    - value         The final value of the fitness function
    """
    
    param_names = get_param_names(params)
    
    NTarget = int(array(data)[:,0].max()+1)
    # N is the number of neurons
    # There are particle_number neurons per target spike train
    N = particle_number * NTarget
    
    initial_param_values = get_initial_param_values(params, N)
    
    vgroup = VectorizedNeuronGroup(model = model, threshold = threshold, reset = reset, 
                 input_name = input_name, input_values = input_values, dt = dt, 
                 slice_number = slice_number,
                 **initial_param_values)
    model_target = kron(arange(NTarget), ones(particle_number))
    cd = CoincidenceCounter(vgroup, data, model_target = model_target, delta = delta)
    
    def fun(X):
        param_values = get_param_values(X, param_names)
        net = Network(vgroup, cd)
        vgroup.set_param_values(param_values)
        reinit_default_clock()
        cd.reinit()
        net.run(vgroup.duration)
        gamma = cd.gamma
        return gamma
    
    X0 = get_matrix(initial_param_values, param_names)
    min_values, max_values = set_constraints(N = N, **params)
    
    X, value = fit(fun, X0, group_size = particle_number, 
                   iterations = iterations,
                   min_values = min_values, max_values = max_values)
    best_params = get_param_values(X, param_names) 
    
    return (Parameters(**best_params), value)


if __name__ == '__main__':
    
    eqs = """
    dV/dt = -V/tau+I : 1
    tau : second
    I : Hz
    """
    NTarget = 1
    tau = .04+.02*rand(NTarget)
    dt = .1*ms
    duration = 400*ms
    I = 120.0/second + 5.0/second * randn(int(duration/dt))

    # Generates data from an IF neuron with tau between 20-40ms
    vgroup = VectorizedNeuronGroup(model = eqs, reset = 0, threshold = 1, 
             input_name = 'I', input_values = I, dt = dt, 
             tau = tau)
    M = SpikeMonitor(vgroup)
    net = Network(vgroup, M)
    net.run(duration)
    data = M.spikes
    
    # Tries to find tau
    params, value = modelfitting(model = eqs, reset = 0, threshold = 1,
                               data = data, 
                               input_name = 'I', 
                               input_values = I,
                               dt = dt,
                               particle_number = 10,
                               iterations = 10,
                               tau = [1*ms, 20*ms, 40*ms, 100*ms],
                               delta = .005
                               )
    
    print "real tau =", tau
    print "computed tau =", params['tau']
    
    
    