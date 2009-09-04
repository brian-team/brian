from brian import *

def vectorized_group(model, reset, threshold, input_name, input_values, dt = .1*ms, 
                     overlap = 200*ms, slice_number = 1, **params):
    """
    Returns a neuron group for simulating a single model with different 
    parameter values and with time parallelization.
    
    Usage example:
    - model           Model equations
    - reset           Model reset
    - threshold       Model threshold 
    - data            A list of spike times (i,t)
    - input_name      The parameter name of the input current in the model equations
    - input_values    The input values
    - dt              Timestep of the input
    - overlap         Overlap between time slices (default 200ms)
    - slice_number    Number of time slices (default 1)
    - **params        Model parameters list : tau=(min,init_min,init_max,max)
    """
    
    values_number = len(params.values()[0]) # Number of parameter values
    N = values_number * slice_number # Total number of neurons
    group = NeuronGroup(N, model = model, threshold = threshold, reset = reset)
    input_length = len(input_values)
    duration = input_length*dt
    
    for param,value in params.iteritems():
        # each neuron is duplicated slice_number times, with the same parameters. 
        # Only the input current changes.
        # new group = [neuron1, ..., neuronN, ..., neuron1, ..., neuronN]
        group.state(param)[:] = kron(ones(slice_number), value)
    # Injects sliced current to each subgroup
    for _ in range(slice_number):
        input_sliced_values = input_values[max(0,input_length/slice_number*_-int(overlap/dt)):input_length/slice_number*(_+1)]
        sliced_subgroup = group.subgroup(values_number)
        sliced_subgroup.set_var_by_array(input_name, TimedArray(input_sliced_values))
    return group

def vectorized_group_test():
    model = """
    dV/dt = -V/tau+I : 1
    tau : second
    I : Hz
    """
    reset = 0
    threshold = 1
    duration = 400*ms
    dt = defaultclock.dt
    overlap = 0*ms
    I = arange(50.0, 100.0, 50.0/int(duration/dt))
    tau = arange(.01,.08,.001)
    vgroup = vectorized_group(model, reset, threshold, 'I', I, overlap = overlap, 
                              slice_number = 1, tau = tau)
    M = SpikeMonitor(vgroup)
    net = Network(vgroup, M)
    net.run(duration)
    raster_plot(M)
    reinit()
    figure()
    vgroup = vectorized_group(model, reset, threshold, 'I', I, overlap = overlap, 
                              slice_number = 4, tau = tau)
    M = SpikeMonitor(vgroup)
    net = Network(vgroup, M)
    net.run(duration/4)
    raster_plot(M)
    
    show()

if __name__ == '__main__':
    vectorized_group_test()
    