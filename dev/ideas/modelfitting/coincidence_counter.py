from brian import *
from vectorized_neurongroup import *

def spikestimes2dict(list):
    """
    Converts a (i,t)-like list into a dictionary of spike trains.
    """
    spikes = {}
    for (i,t) in list:
        if i not in spikes.keys():
            spikes[i] = [t]
        else:
            spikes[i].append(t)
    for key in spikes.keys():
        spikes[key].sort()
        spikes[key] = array(spikes[key])
    return spikes

def dict2spiketimes(trains):
    """
    Converts a dictionary of spike trains into a (i,t)-like list.
    """
    spiketimes = []
    duration = 0.0
    for (i,train) in trains.iteritems():
        for t in train:
            spiketimes.append((i, t*second))
            if (t > duration):
                duration = t
    duration += .01
    spiketimes.sort(cmp = lambda x,y: int(x[1]>y[1])*2-1)
    return spiketimes

def slice_data(data = None, slice_number = 1, duration = None):
        """
        Slices several spike trains into slice_number time slices each.
        """
        if slice_number == 1:
            return array(data)
        n = array(data)[:,0].max()
        sliced_data = []
        slice_length = duration/slice_number
        for i,t in data:
            sliced_i = i*slice_number + int(t/slice_length)
            sliced_t = t - int(t/slice_length)*slice_length
            sliced_data.append((sliced_i, sliced_t))
        return sliced_data

#data=[(0,0),(0,1),(0,1.4),(0,1.9)]
#print slice_data(data = data, slice_number = 2, duration = 2)

class CoincidenceCounter(SpikeMonitor):
    
    def __init__(self, source, data, model_target, delta = .004):
        """
        Computes in an online fashion the number of coincidences between model 
        spike trains and some data spike trains.
        
        Usage example:
        cd = CoincidenceCounter(group, data, model_target = [0])
        
        Inputs:
        - source        The NeuronGroup object
        - data          The data as an (i,t)-like list.
        - model_target  The target train index associated to each model neuron.
                        It is a list of size NTarget.
        - delta         The half-width of the time window
        
        Outputs:
        - cd.coincidences is a N-long vector with the coincidences of each neuron
        versus its linked target spike train.
        """
        source.set_max_delay(0)
        self.source = source
        self.delay = 0
        
        # Adapts data if there are several time slices
        if isinstance(source, VectorizedNeuronGroup):
            self.data = slice_data(data = data, slice_number = source.slice_number, duration = source.total_duration)
        else:
            self.data = data
        
        self.NModel = len(source)
        self.NTarget = int(array(self.data)[:,0].max()+1)
        
        # Adapts model_target if there are several time slices
        model_target = array(model_target)
        if isinstance(source, VectorizedNeuronGroup):
            self.model_target = []
            for i in range(source.slice_number):
                self.model_target += list(model_target*source.slice_number + i)
            self.model_target = array(self.model_target)
        else:
            self.model_target = array(model_target)
        
        self.delta = delta
        self.dt = defaultclock._dt
        
        self._coincidences = zeros(self.NModel)
        
        self.prepare_online_computation()
        
    def propagate(self, spiking_neurons):
        '''
        Is called during the simulation, each time a neuron spikes. Spiking_neurons is the list of 
        the neurons which have just spiked.
        self.close_target_spikes is the list of the closest target spike for each target train in the current bin self.current_bin
        '''
        if (self.current_bin < len(self.all_bins)-1):
            if defaultclock._t > self.all_bins[self.current_bin+1]:
                self.current_bin += 1
                self.close_target_spikes = self.close_target_spikes_matrix[:,self.current_bin]
        
        if (len(spiking_neurons) > 0):
            close_target_spikes = (self.close_target_spikes >= 0)
            target_spikes = nonzero(close_target_spikes)[0]
            if (len(target_spikes) > 0):
                for i in spiking_neurons:
                    target_spikes2 = nonzero(close_target_spikes & (self.close_target_spikes > self.last_target_spikes[i]))[0]
                    j = self.model_target[i]
                    if j in target_spikes2:
                        self._coincidences[i] += 1
                        self.last_target_spikes[i] = self.close_target_spikes[j]

    def prepare_online_computation(self):
        '''
        Preparation step for the online algorithm, called just once (depends only on the target trains)
        Shouldn't be called at each optimization iteration
        '''
        self.reinit()
        self.compute_all_bins()
        self.compute_close_target_spikes()
    
    def reinit(self):
        '''
        Reinitializes the object after a run.
        Used to compute the gamma factor with different model trains, but identic target trains 
        (the results of the preparation step are kept in the object)
        '''
        self.close_target_spikes    = -1 * ones(self.NTarget)
        self.last_target_spikes     = -1 * ones(self.NModel)
        self.current_bin            = -1
        self._coincidences           = zeros(self.NModel)
    
    def compute_all_bins(self):
        '''
        Called during the online preparation step.
        Computes self.all_bins : the reunion of target_train+/- delta for all target trains
        '''
        all_times = array([t for i,t in self.data])
        all_times = floor(all_times/self.dt)*self.dt # HACK : forces the precision of 
                                                   # data spike trains to be the same
                                                   # as defaultclock.dt
        self.all_bins = sort(list(all_times - self.delta - self.dt/10) + list(all_times + self.delta + self.dt/10))
        
    def compute_close_target_spikes(self):
        '''
        Called during the online preparation step.
        Computes the closest target spikes for each target train.
        
        The result is a NTarget*len(all_bins) array,
        result[i,j] is the index of the closest target spike index of target train i in bin j, only if it is at a distance < delta in the bin
        It is -1 if there is no target spike within +- delta in the bin.
        
        For example, close_target_spikes_matrix is always [-1,0,-1,1,-1,2,-1,3...] when there is only one target train :
        The first bin is before the first target spike - delta : no close target spike in that bin
        The second bin is between the first target spike +- delta : the closest target spike is the number 0 (the first spike of the target train)
        etc.
        '''
        self.close_target_spikes_matrix = -1*ones((self.NTarget, len(self.all_bins)))
        all_bins_centers = (self.all_bins[0:-1] + self.all_bins[1:])/2
        target_trains = spikestimes2dict(self.data)
        for j,b in enumerate(all_bins_centers):
            for i,train in target_trains.iteritems():
                ind = nonzero(abs(train-b) <= self.delta+self.dt/10)[0]
                if (len(ind)>0):
                    self.close_target_spikes_matrix[i, j] = ind[0]

    def get_coincidences(self):
        if isinstance(self.source, VectorizedNeuronGroup):
            return self._coincidences.reshape((self.source.slice_number,self.source.neuron_number)).sum(axis=0)
        else:
            return self._coincidences
        
    def set_coincidences(self, value):
        self._coincidences = value
        
    coincidences = property(get_coincidences, set_coincidences)