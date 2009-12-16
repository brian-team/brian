from brian import *
from clustertools import *
from fittingparameters import *
from brian.utils.particle_swarm import *
try:
    import pycuda
    from gpu_modelfitting import GPUModelFitting
    can_use_gpu = True
except ImportError:
    can_use_gpu = False
    
class modelfitting_worker(object):
    def __init__(self, shared_data, use_gpu):

        neurons = shared_data['neurons']
        model = shared_data['model']
        threshold = shared_data['threshold']
        reset = shared_data['reset']
        input = shared_data['input']
        I_offset = shared_data['I_offset']
        dt = shared_data['dt']
        duration = shared_data['duration']
        onset = shared_data['onset']
        spiketimes = shared_data['spiketimes']
        spiketimes_offset = shared_data['spiketimes_offset']
        spikedelays = shared_data['spikedelays']
        initial_values = shared_data['initial_values']
        delta = shared_data['delta']
        includedelays = shared_data['includedelays']
        params = shared_data['params']
#        neurons, model, threshold, reset, input, I_offset, dt, duration, onset,
#        spiketimes, spiketimes_offset, spikedelays, initial_values, delta, includedelays, **params

        self.duration = duration
        self.neurons = neurons
        self.includedelays = includedelays
        # Loads parameters
        self.fp = FittingParameters(includedelays = includedelays, **params)
        self.param_names = fp.param_names

        self.group = NeuronGroup(neurons, model = model, reset = reset, threshold = threshold)
        if initial_values is not None:
            for param, value in initial_values.iteritems():
                self.group.state(param)[:] = value
    
        # INJECTS CURRENT
        # Injects current in consecutive subgroups, where I_offset have the same value
        # on successive intervals
        k = -1
        for i in hstack((nonzero(diff(I_offset))[0], len(I_offset)-1)):
            I_offset_subgroup_value = I_offset[i]
            I_offset_subgroup_length = i-k
            # DEBUG
    #                print I_offset_subgroup_value, I_offset_subgroup_length
            sliced_subgroup = self.group.subgroup(I_offset_subgroup_length)
            input_sliced_values = input[I_offset_subgroup_value:I_offset_subgroup_value+total_steps]
            sliced_subgroup.set_var_by_array(input_var, TimedArray(input_sliced_values, clock=group.clock))
            k = i  
            
        self.cc = CoincidenceCounterBis(self.group, spiketimes, spiketimes_offset, onset = onset, delta = delta)
        
    def process(self, X):
        # Gets the parameter values contained in the matrix X, excepted spike delays values
        if self.includedelays:
            param_values = self.fp.get_param_values(X[0:-1,:], includedelays = False)
        else:
            param_values = self.fp.get_param_values(X, includedelays = False)
        # Sets the parameter values in the NeuronGroup object
        for param, value in param_values.iteritems():
            self.group.state(param)[:] = value
        # Sets the spike delay values
        if includedelays:
            self.cc.spikedelays = X[-1,:]
        # Reinitializes the simulation objects
        reinit_default_clock()
        self.cc.reinit()
        net = Network(self.group, self.cc)
        # LAUNCHES the simulation on the CPU
        net.run(self.duration)
    
        return self.cc.coincidences, self.cc.model_length



def modelfitting(model = None, reset = None, threshold = None, data = None, 
                 input_var = 'I', input = None, dt = None,
                 verbose = True, particles = 100, slices = 1, overlap = None,
                 iterations = 10, delta = None, initial_values = None, stepsize = 100*ms,
                 use_gpu = None, includedelays = True,
                 **params):
    
    
    
    
    shared_data = dict(
        neurons = neurons,
        model = model,
        threshold = threshold, 
        reset = reset, 
        input = input, 
        I_offset = I_offset, 
        dt = dt, 
        duration = duration, 
        onset = onset,
        spiketimes = spiketimes,
        spiketimes_offset = spiketimes_offset, 
        spikedelays = spikedelays, 
        initial_values = initial_values, 
        delta = delta, 
        includedelays = includedelays,
        params = params
    )
    manager = ClusterManager(work_class, shared_data)
    
    # TODO: construct X_list (one
    
    results = manager.process_jobs(X_list)
    manager.finished()
    print results

if __name__=='__main__':
    def get_model():
        model = Equations('''
            dV/dt=(R*I-V)/tau : 1
            I : 1
            R : 1
            tau : second
        ''')
        reset = 0
        threshold = 1
        return model, reset, threshold
    
    def get_data(**params):
        # DATA GENERATION
        # at the end, data should be an (i,t) list
        group = NeuronGroup(N = ntrials, model = model, reset = reset, threshold = threshold)
        for param, value in params.iteritems():
            group.state(param)[:] = value
        group.I = TimedArray(input, start = 0*second, dt = defaultclock.dt)
        
        M = SpikeMonitor(group)
        StM = StateMonitor(group, 'V', record = True)
        net = Network(group, M, StM)
        
        reinit_default_clock()
        net.run(duration)
        
        data_spikes = M.spikes
        data_values = StM.values
        
        reinit_default_clock()
        return data_spikes, data_values
    
    def get_current():
        # CURRENT GENERATION
        # at the end, I should be the list of the I values
        # and dt the timestep
        dt = .1*ms
        n = int(duration/dt)
        I = .48+.8*randn(n)
        return I, dt
    
    slices = 1
    ntrials = 2
    duration = 500*ms
    overlap = 300*ms
    group_size = 10000 # number of neurons per target train
    delta = .5*ms
    iterations = 1
    
    tau0 = array([22*ms, 28*ms])
    R0 = array([2.1, 2.5])
    
    model, reset, threshold = get_model()
    input, dt = get_current()
    data_spikes, data_values = get_data(R=R0, tau=tau0)
    
    i, t = zip(*data_spikes)
    i = array(i)
    t = array(t)
    for j in range(ntrials):
        s = sort(t[i==j])
        print "Train %d" % j
        print array(1000*s, dtype=int)/1000.0
        print
    
    import time
    start = time.clock()
    params, gamma = modelfitting(model, reset, threshold, data_spikes, 
                    input = input, dt = dt,
                    verbose = True, particles = group_size, slices = slices, overlap = overlap,
                    iterations = iterations, delta = delta, 
                    initial_values = {'V': 0},
                    use_gpu = False,
                    includedelays = True,
                    R = [2.0, 2.0, 2.6, 2.6],
                    tau = [20*ms, 20*ms, 30*ms, 30*ms])
    end = time.clock()
    
    print params
    
    print 'Total time: %.3f seconds' % (end-start)


