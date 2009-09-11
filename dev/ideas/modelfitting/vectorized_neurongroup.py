from brian import *
from IPython.DPyGetOpt import ArgumentError

class VectorizedNeuronGroup(NeuronGroup):
    """
    Neuron group defining a single model with different 
    parameter values and with time parallelization.
    
    Inputs:
    - model           Model equations
    - reset           Model reset
    - threshold       Model threshold 
    - data            A list of spike times (i,t)
    - input_name      The parameter name of the input current in the model equations
    - input_values    The input values
    - dt              Timestep of the input
    - overlap         Overlap between time slices
    - slice_number    Number of time slices (default 1)
    - **param_values  Model parameters values
    """
    
    def __init__(self, model = None, threshold = None, reset = NoReset(), 
                 input_name = 'I', input_values = None, dt = .1*ms, 
                 overlap = 0*ms, slice_number = 1, **param_values):
        
        if slice_number == 1:
            overlap = 0*ms
        values_number = len(param_values.values()[0]) # Number of parameter values
        for param, value in params.iteritems():
            if not(len(value) == values_number):
                raise AttributeError, 'The parameters must have the same number of values'
        
        N = values_number * slice_number # Total number of neurons
        NeuronGroup.__init__(self, N = N, model = model, threshold = threshold, reset = reset)
        input_length = len(input_values)
        
        self.neuron_number = values_number
        self.slice_number = slice_number
        self.overlap = overlap
        self.total_duration = input_length*dt
        self.duration = self.total_duration/slice_number+overlap
        
        if overlap >= input_length*dt/slice_number:
            raise AttributeError, 'Overlap should be less than %.2f' % input_length*dt/slice_number
        
        self.set_param_values(param_values)
        
        # Injects sliced current to each subgroup
        for _ in range(slice_number):
            if _ == 0:
                input_sliced_values = concatenate((zeros(int(overlap/dt)),input_values[0:input_length/slice_number]))
            else:
                input_sliced_values = input_values[input_length/slice_number*_-int(overlap/dt):input_length/slice_number*(_+1)]
            sliced_subgroup = self.subgroup(values_number)
            sliced_subgroup.set_var_by_array(input_name, TimedArray(input_sliced_values))
        
    def set_param_values(self, param_values):
        for param,value in param_values.iteritems():
            # each neuron is duplicated slice_number times, with the same parameters. 
            # Only the input current changes.
            # new group = [neuron1, ..., neuronN, ..., neuron1, ..., neuronN]
            self.state(param)[:] = kron(ones(slice_number), value)
        
        
        
        