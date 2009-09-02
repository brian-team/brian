from brian import *
from CoincidenceCounter import *

def modelfitting(eqs, reset, threshold, data, input, dt = 0.1*ms,
                 verbose = False, timeslices = 1, **params):
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
    - eqs           Neuron model equations
    - reset         Neuron model reset
    - threshold     Neuron model threshold
    - data          A list of spike times (i,t)
    - input         The input current (a list of values)
    - dt            Timestep of the input
    - verbose       Print iterations?
    - params        Model parameters list : tau=(min,init_min,init_max,max)
    - timeslices    Number of time slices, 1 by default
    
    Outputs:
    - params        The parameter values found by the optimization process
    - value         The final value of the fitness function
    """
    
    model = Equations(eqs)
    # TODO
    
    return (Parameters(tau = rand(3)), .8)
    
    