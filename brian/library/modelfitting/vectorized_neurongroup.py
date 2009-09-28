from brian.neurongroup import *
from brian.reset import *
from brian.timedarray import TimedArray 
from numpy import kron, ones, zeros, concatenate

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
    - input    The input values
    - overlap         Overlap between time slices
    - slices    Number of time slices (default 1)
    - **param_values  Model parameters values
    """
    
    def __init__(self, model = None, threshold = None, reset = NoReset(), 
                 input_var = 'I', input = None,
                 overlap = None, slices = 1, **param_values):
        
        if (slices == 1) or (overlap is None):
            overlap = 0*ms
        values_number = len(param_values.values()[0]) # Number of parameter values
        for param, value in param_values.iteritems():
            if not(len(value) == values_number):
                raise AttributeError, 'The parameters must have the same number of values'
        
        N = values_number * slices # Total number of neurons
        NeuronGroup.__init__(self, N = N, model = model, threshold = threshold, reset = reset)
        dt=self.clock.dt
        input_length = len(input)
        
        self.neuron_number = values_number
        self.slices = slices
        self.overlap = overlap
        self.total_duration = input_length*dt
        self.duration = self.total_duration/slices+overlap
        
        if overlap >= input_length*dt/slices:
            raise AttributeError, 'Overlap should be less than %.2f' % input_length*dt/slices

        self.set_param_values(param_values)
        
        # Injects sliced current to each subgroup
        for _ in range(slices):
            if _ == 0:
                input_sliced_values = concatenate((zeros(int(overlap/dt)),input[0:input_length/slices]))
            else:
                input_sliced_values = input[input_length/slices*_-int(overlap/dt):input_length/slices*(_+1)]
            sliced_subgroup = self.subgroup(values_number)
            sliced_subgroup.set_var_by_array(input_var, TimedArray(input_sliced_values))
        
    def set_param_values(self, param_values):
        for param,value in param_values.iteritems():
            # each neuron is duplicated slice_number times, with the same parameters. 
            # Only the input current changes.
            # new group = [neuron1, ..., neuronN, ..., neuron1, ..., neuronN]
            self.state(param)[:] = kron(ones(self.slices), value)
        