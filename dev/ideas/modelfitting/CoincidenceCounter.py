from brian import *

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
        
        self.NTarget = self.data[:,0].max()+1
        self.NModel = len(source)
        
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


def CoincidenceCounterTest():
    eqs = 'dV/dt = -V/tau+I : 1'
    tau = 20*ms
    I = 150/second
    n = 10
    isi = -tau*log(1-1/(tau*I)) # we compute the isi 
    duration = n*isi+.002*second

    group = NeuronGroup(1, 
                        model = eqs,
                        reset = 0,
                        threshold = 1)
    data = [(0,floor(10000*t)/10000*second) for t in cumsum(isi*ones(n))] # we compute the predicted spike train
    cd = CoincidenceCounter(group, data, model_target = [0])
    M = SpikeMonitor(group)
    run(duration)
    
    print "Model spikes :"
    print M.spikes
    print "Predicted spikes :"
    print data
    print "Coincidences :"
    print "%d/%d" % (cd.coincidences, n)
    if cd.coincidences == n:
        print 'Coincidence counter test : OK'
        return True
    else:
        print 'Coincidence counter test : Failed'
        return False


if __name__ == '__main__':
    CoincidenceCounterTest()