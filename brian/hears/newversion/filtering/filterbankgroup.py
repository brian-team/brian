from brian import StateUpdater, NeuronGroup, Equations, Clock, network_operation

__all__ = ['FilterbankGroup']

class FilterbankGroup(NeuronGroup):
    '''
    Allows a Filterbank object to be used as a NeuronGroup
    
    Initialised as a standard :class:`NeuronGroup` object, but with two
    additional arguments at the beginning, and no ``N`` (number of neurons)
    argument.  The number of neurons in the group will be the number of
    channels in the filterbank. (TODO: add reference to interleave/serial
    channel stuff here.)
    
    ``filterbank``
        The Filterbank object to be used by the group. In fact, any Bufferable
        object can be used.
    ``targetvar``
        The target variable to put the filterbank output into.
        
    One additional keyword is available beyond that of :class:`NeuronGroup`:
    
    ``buffersize=32``
        The size of the buffered segments to fetch each time. The efficiency
        depends on this in an unpredictable way, larger values mean more time
        spent in optimised code, but are worse for the cache. In many cases,
        the default value is a good tradeoff. Values can be given as a number
        of samples, or a length of time in seconds.
        
    Note that if you specify your own :class:`Clock`, it should have
    1/dt=samplerate.
    '''
    
    def __init__(self, filterbank, targetvar, *args, **kwds):
        self.targetvar = targetvar
        self.filterbank = filterbank
        filterbank.buffer_init()

        # update level keyword
        kwds['level'] = kwds.get('level', 0)+1
    
        # Sanitize the clock - does it have the right dt value?
        if 'clock' in kwds:
            if int(1/kwds['clock'].dt)!=int(filterbank.samplerate):
                raise ValueError('Clock should have 1/dt=samplerate')
        else:
            kwds['clock'] = Clock(dt=1/filterbank.samplerate)        
        
        buffersize = kwds.pop('buffersize', 32)
        if not isinstance(buffersize, int):
            buffersize = int(buffersize*self.samplerate)
        self.buffersize = buffersize
        self.buffer_pointer = buffersize
        self.buffer_start = -buffersize
        
        NeuronGroup.__init__(self, filterbank.nchannels, *args, **kwds)
        
        @network_operation(when='start', clock=self.clock)
        def apply_filterbank_output():
            if self.buffer_pointer>=self.buffersize:
                self.buffer_pointer = 0
                self.buffer_start += self.buffersize
                self.buffer = self.filterbank.buffer_fetch(self.buffer_start, self.buffer_start+self.buffersize)
            setattr(self, targetvar, self.buffer[self.buffer_pointer, :])
            self.buffer_pointer += 1
        
        self.contained_objects.append(apply_filterbank_output)
        
    def reinit(self):
        NeuronGroup.reinit(self)
        self.filterbank.buffer_init()
        self.buffer_pointer = self.buffersize
        self.buffer_start = -self.buffersize
