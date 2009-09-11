from brian import *
from coincidence_counter import *


# TODO
def get_initial_param_values(params, N):
    """
    Samples random initial param values around default values
    """
    pass

def get_param_names(params):
    """
    Returns the sorted list of parameter names.
    """
    param_names = sort(params.keys())
    return param_names

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


# TODO
def fit(fun, X0, group_size):
    """
    Maximizes a function starting from initial values X0
    if y=fun(x), 
        x is a D*(group_size*Ntarget) matrix
        y is a group_size*Ntarget vector
    fit maximizes fun independently over Ntarget subgroups
    fit returns a D*Ntarget matrix.
    """
    pass


def modelfitting(eqs, reset, threshold, data, input_name, input_values, dt = 0.1*ms,
                 timeslices = 1, verbose = False, particle_number = 1, slice_number = 1,
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
    - **params        Model parameters list : tau=(min,init_min,init_max,max)
    - verbose       Print iterations?
    - particle_number
                    Number of particles in the particle swarm algorithm
    - slice_number  Number of time slices, 1 by default
    
    Outputs:
    - params        The parameter values found by the optimization process
    - value         The final value of the fitness function
    """
    
    param_names = get_param_names(params)
    
    Ntarget = array(data)[:,0].max()+1
    # N is the number of neurons
    # There are particle_number neurons per target spike train
    N = particle_number * Ntarget
    
    initial_param_values = get_initial_param_values(params, N)
    
    vgroup = VectorizedNeuronGroup(model = model, threshold = threshold, reset = reset, 
                 input_name = input_name, input_values = input_values, dt = dt, 
                 params = initial_param_values)
    model_target = kron(arange(vgroup.neuron_number), ones(particle_number))
    cd = CoincidenceCounter(vgroup, data, model_target = model_target)
    
    def fun(X):
        param_values = get_param_values(X, param_names)
        vgroup.set_param_values(param_values)
        cd.reinit()
        run(vgroup.duration)
        return cd.gamma
    
    X0 = get_matrix(initial_param_values, param_names)
    fit(fun, X0, group_size = particle_number)
    
    return (Parameters(tau = rand(3)), .8)
    
    