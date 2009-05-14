from brian import *

#class RecentStateMonitor(StateMonitor):
#    '''
#    StateMonitor that records only the most recent fixed amount of time.
#    
#    Works in the same way as a :class:`StateMonitor` except that it has one
#    additional initialiser keyword ``duration`` which gives the length of
#    time to record values for, the ``record`` keyword defaults to ``True``
#    isntead of ``False``, and there are some different or additional
#    attributes:
#    
#    ``values``, ``values_``,
#    ``times``, ``times_``
#        These will now return at most the most recent values over an
#        interval of maximum time ``duration``. These arrays are copies,
#        so for faster access use ``unsorted_values``, etc.
#    ``unsorted_values``, ``unsorted_values_``,
#    ``unsorted_times``, ``unsorted_times_``
#        The raw versions of the data, the associated times may not be
#        in sorted order and if ``duration`` hasn't passed, not all the
#        values will be meaningful.
#    ``current_time_index``
#        Says which time index the next values to be recorded will be stored
#        in, varies from 0 to M-1.
#    ``has_looped``
#        Whether or not the ``current_time_index`` has looped from M back to
#        0 - can be used to tell whether or not every value in the
#        ``unsorted_values`` array is meaningful or not (they will only all
#        be meaningful when ``has_looped==True``, i.e. after time ``duration``).
#    
#    The ``__getitem__`` method also returns values in sorted order.
#    
#    To plot, do something like::
#    
#        plot(M.times, M.values[:, i])
#    '''
#    def __init__(self, P, varname, duration=5*ms, clock=None, record=True, timestep=1, when='end'):
#        StateMonitor.__init__(self, P, varname, clock=clock, record=record, timestep=timestep, when=when)
#        self.num_duration = int(duration/(timestep*self.clock.dt))+1
#        if record is False:
#            self.record_size = 0
#        elif record is True:
#            self.record_size = len(P)
#        elif isinstance(record, int):
#            self.record_size = 1
#        else:
#            self.record_size = len(record)
#        self._values = zeros((self.num_duration, self.record_size))
#        self._times = zeros(self.num_duration)
#        self.current_time_index = 0
#        self.has_looped = False
#        
#    def __call__(self):
#        V = self.P.state_(self.varname)
#        self._mu += V
#        self._sqr += V*V
#        if self.record is not False and self.curtimestep==self.timestep:
#            i = self._recordstep
#            if self.record is not True:
#                self._values[self.current_time_index, :] = V[self.record]
#            else:
#                self._values[self.current_time_index, :] = V
#            self._times[self.current_time_index] = self.clock.t
#            self._recordstep += 1
#            self.current_time_index = (self.current_time_index+1)%self.num_duration
#            if self.current_time_index==0: self.has_looped = True
#        self.curtimestep -= 1
#        if self.curtimestep==0: self.curtimestep = self.timestep
#        self.N += 1
#    
#    def __getitem__(self,i):
#        timeinds = self.sorted_times_indices()
#        if self.record is False:
#            raise IndexError('Neuron ' + str(i) + ' was not recorded.')
#        if self.record is not True:
#            if isinstance(self.record,int) and self.record!=i or (not isinstance(self.record,int) and i not in self.record):
#                raise IndexError('Neuron ' + str(i) + ' was not recorded.')
#            try:
#                return QuantityArray(self._values[timeinds, self.recordindex[i]])*self.unit
#            except:
#                if i==self.record:
#                    return QuantityArray(self._values[timeinds, 0])*self.unit
#                else:
#                    raise
#        elif self.record is True:
#            return QuantityArray(self._values[timeinds, i])*self.unit
#        
#    def getvalues(self):
#        return safeqarray(self._values, units=self.unit)
#
#    def getvalues_(self):
#        return self._values
#    
#    def sorted_times_indices(self):
#        if not self.has_looped:
#            return arange(self.current_time_index)
#        return argsort(self._times)
#
#    def get_sorted_times(self):
#        return safeqarray(self._times[self.sorted_times_indices()], units=second)
#    
#    def get_sorted_times_(self):
#        return self._times[self.sorted_times_indices()]
#    
#    def get_sorted_values(self):
#        return safeqarray(self._values[self.sorted_times_indices(), :], units=self.unit)
#    
#    def get_sorted_values_(self):
#        return self._values[self.sorted_times_indices(), :]
#
#    times  = property(fget=get_sorted_times)
#    times_ = property(fget=get_sorted_times_)
#    values = property(fget=get_sorted_values)
#    values_= property(fget=get_sorted_values_)
#    unsorted_times  = property(fget=lambda self:QuantityArray(self._times))
#    unsorted_times_ = property(fget=lambda self:array(self._times))
#    unsorted_values = property(fget=getvalues)
#    unsorted_values_= property(fget=getvalues_)
#    
#    def reinit(self):
#        self._values[:] = 0
#        self._times[:] = 0
#        self.current_time_index = 0
#        self.N = 0
#        self._recordstep = 0
#        self._mu=zeros(len(self.P))
#        self._sqr=zeros(len(self.P))
#        self.has_looped = False

if __name__=='__main__':
    G = NeuronGroup(1, 'dV/dt = xi/(10*ms)**0.5 : 1')
    MR = RecentStateMonitor(G, 'V')
    M = StateMonitor(G, 'V', record=True)
    run(7*ms)
    M.plot()
    plot(MR.times, MR[0]+0.1)
    MR.plot()
    show()