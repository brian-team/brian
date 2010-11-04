'''
The Bufferable class serves as a base for all the other Brian.hears classes
'''
from numpy import zeros, empty

class Bufferable(object):
    '''
    Base class for Brian.hears classes
    
    Defines a buffering interface of two methods:
    
    ``buffer_init()``
        Initialise the buffer, should set the time pointer to zero and do
        any other initialisation that the object needs.
        
    ``buffer_fetch(start, end)``
        Fetch the next samples ``start:end`` from the buffer. Value returned
        should be an array of shape ``(end-start, nchannels)``. Can throw an
        ``IndexError`` exception if it is outside the possible range.
        
    In addition, bufferable objects should define attributes:
    
    ``nchannels``
        The number of channels in the buffer.
        
    ``samplerate``
        The sample rate in Hz.
        
    By default, the class will define a default buffering mechanism which can
    easily be extended. To extend the default buffering mechanism, simply
    implement the method:
    
    ``buffer_fetch_next(samples)``
        Returns the next ``samples`` from the buffer.
    
    The default methods for ``buffer_init()`` and ``buffer_fetch()`` will
    define a buffer cache which will get larger if it needs to to accommodate
    a ``buffer_fetch(start, end)`` where ``end-start`` is larger than the
    current cache. The following attributes will automatically be maintained:
    
    ``self.cached_buffer_start``, ``self.cached_buffer_end``
        The start and end of the cached segment of the buffer
    
    ``self.cached_buffer_output``
        An array of shape ``((cached_buffer_end-cached_buffer_start, nchannels)``
        with the current cached segment of the buffer. Note that this array can
        change size.
    '''
    def buffer_fetch(self, start, end):
        # optimisations for the most typical cases, which are when start:end is
        # the current cached segment, or when start:end is the next cached
        # segment of the same size as the current one
        if start==self.cached_buffer_start and end==self.cached_buffer_end:
            return self.cached_buffer_output
        if start==self.cached_buffer_end and end-start==self.cached_buffer_output.shape[0]:
            self.cached_buffer_output = self.buffer_fetch_next(end-start)
            self.cached_buffer_start = start
            self.cached_buffer_end = end
            return self.cached_buffer_output
        # handle bad inputs
        if end<start:
            raise IndexError('Buffer end should be larger than start.')
        if start<self.cached_buffer_start:
            raise IndexError('Attempted to fetch output that has disappeared from the buffer.')
        # If the requested segment of the buffer is entirely within the cache,
        # just return it.
        if end<=self.cached_buffer_end:
            bstart = start-self.cached_buffer_start
            bend = end-self.cached_buffer_start
            return self.cached_buffer_output[bstart:bend, :]
        # Otherwise we need to fetch some new samples.
        samples = end-self.cached_buffer_end
        newsegment = self.buffer_fetch_next(samples)
        # otherwise we have an overlap situation - I guess this won't really
        # happen very often but it has to be handled correctly in case.
        new_end = end
        # if end-start is longer than the size of the current cached buffer, we
        # will have to increase the size of the cache
        new_start = min(new_end-self.cached_buffer_output.shape[0], start)
        new_size = new_end-new_start
        new_output = empty((new_size, self.nchannels))
        new_output[:new_size-samples, :] = self.cached_buffer_output[samples-new_size:, :]
        new_output[new_size-samples:, :] = newsegment
        self.cached_buffer_output = new_output
        self.cached_buffer_start = new_start
        self.cached_buffer_end = new_end
        return new_output[start-self.cached_buffer_start:, :]
    
    def buffer_init(self):
        self.cached_buffer_output = zeros((0, self.nchannels))
        self.cached_buffer_start = 0
        self.cached_buffer_end = 0
    
    def buffer_fetch_next(self, samples):
        raise NotImplementedError
    
    nchannels = NotImplemented
    samplerate = NotImplemented
