from brian import *

#def spikestimes2dict(list):
#    """
#    Converts a (i,t)-like list to a dictionary of spike trains.
#    """
#    spikes = {}
#    for (i,t) in list:
#        if i not in spikes.keys():
#            spikes[i] = [t]
#        else:
#            spikes[i].append(t)
#    for key in spikes.keys():
#        spikes[key].sort()
#        spikes[key] = array(spikes[key])
#    return spikes

class CoincidenceCounter(SpikeMonitor):
    
    def __init__(self, source, data, model_target, delta = .004):
        """
        Computes in an online fashion the gamma factor of model spike trains against
        some data spike trains.
        
        Inputs:
        - source        The NeuronGroup object
        - data          The data as a (i,t)-like list.
        - model_target  The target train index associated to each model neuron.
                        It is a list of size NTarget.
        - delta         The half-width of the time window
        """
        source.set_max_delay(0)
        self.source = source
        self.delay = 0
        self.data = array(data)
        
        self.NTarget = data[:,0].max()+1
        self.NModel = source.N
        
        self.model_target = model_target
        self.delta = delta
        self.dt = defaultclock._dt
        
        self.coincidences = zeros(self.NModel)
        
        self.prepare_online_computation()
        
    def propagate(self, spiking_neurons):
        '''
        Is called during the simulation, each time a neuron spikes. Spiking_neurons is the list of 
        the neurons which have just spiked.
        self.close_target_spikes is the list of the closest target spike for each target train in the current bin self.current_bin
        '''
        
        if defaultclock._t in self.all_bins:
            if (self.current_bin < len(self.all_bins)-1):
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
                        self.coincidences[i] += 1
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
        self.coincidences           = zeros(self.NModel)
    
    def compute_all_bins(self):
        '''
        Called during the online preparation step.
        Computes self.all_bins : the reunion of target_train+/- delta for all target trains
        '''
        all_times = array([t for i,t in self.data])
        all_times = int(all_times/self.dt)*self.dt # HACK : forces the precision of 
                                                   # data spike trains to be the same
                                                   # as defaultclock.dt
        all_bins = sort(list(all_times - self.delta - self.dt/10) + list(all_times + self.delta + self.dt/10))
        self.all_bins = all_bins

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
        close_target_spikes_matrix = -1*ones((self.NTarget, len(self.all_bins)))
        dt = self.dt
        for j in range(len(self.all_bins)):
            for i in range(self.NTarget):
                t = self.target_trains[i]
                if (j+1 < len(self.all_bins)):
                    b = (self.all_bins[j]+self.all_bins[j+1])/2
                    ind = nonzero(abs(t-b) <= self.delta+dt/10)[0]
                    if (len(ind)>0):
                        close_target_spikes_matrix[i, j] = ind[0]
                else: # Last bin
                    close_target_spikes_matrix[i, j] = -1
        self.close_target_spikes_matrix = close_target_spikes_matrix
    

