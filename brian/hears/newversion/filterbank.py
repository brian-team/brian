from brian import *
from scipy import signal, weave, random
from bufferable import Bufferable

__all__ = ['Filterbank', 'FunctionFilterbank', 'SumFilterbank',
           'DoNothingFilterbank']

class Filterbank(Bufferable):
    '''
    Generalised filterbank object
    
    This class is a base class not designed to be instantiated. A Filterbank
    object should define the interface of :class:`Bufferable`, as well as
    defining a ``source`` attribute. This is normally a :class:`Bufferable`
    object, but could be an iterable of sources (for example, for filterbanks
    that mix or add multiple inputs).
    
    The ``buffer_fetch_next(samples)`` method has a default implementation
    that fetches the next input, and calls the ``buffer_apply(input)``
    method on it, which can be overridden by a derived class. This is typically
    the easiest way to implement a new filterbank. Filterbanks with multiple
    sources will need to override this default implementation.
    
    There is a default ``__init__`` method that can be called by a derived class
    that sets the ``source``, ``nchannels`` and ``samplerate`` from that of the
    ``source`` object. For multiple sources, the default implementation will
    check that each source has the same number of channels and samplerate and
    will raise an error if not.
    
    There is a default ``buffer_init()`` method that calls ``buffer_init()`` on
    the ``source`` (or list of sources).
    
    Also defines arithmetical operations for +, -, *, / where the other
    operand can be a filterbank or scalar. TODO: add more arithmetical ops?
    e.g. could have **, __abs__.
    '''
    
    source = NotImplemented
    
    def __init__(self, source):
        if isinstance(source, Bufferable):
            self.source = source
            self.nchannels = source.nchannels
            self.samplerate = source.samplerate
        else:
            self.nchannels = source[0].nchannels
            self.samplerate = source[0].samplerate
            for s in source:
                if s.nchannels!=self.nchannels:
                    raise ValueError('All sources must have the same number of channels.')
                if int(s.samplerate)!=int(self.samplerate):
                    raise ValueERror('All sources must have the same samplerate.')
            self.source = source            

    def buffer_init(self):
        Bufferable.buffer_init(self)
        if isinstance(self.source, Bufferable):
            self.source.buffer_init()
        else:
            for s in self.source:
                s.buffer_init()

    def buffer_apply(self, input):
        raise NotImplementedError

    def buffer_fetch_next(self, samples):
        start = self.cached_buffer_end
        end = start+samples
        input = self.source.buffer_fetch(start, end)
        return self.buffer_apply(input)
    
    def __add__ (self, other):
        if isinstance(other, Bufferable):
            return SumFilterbank((self, other))
        else:
            func = lambda x: other+x
            return FunctionFilterbank(self, func)
    __radd__ = __add__
        
    def __sub__ (self, other):
        if isinstance(other, Bufferable):
            return SumFilterbank((self, other), (1, -1))
        else:
            func = lambda x: x-other
            return FunctionFilterbank(self, func)
        
    def __rsub__ (self, other):
        # Note that __rsub__ should return other-self
        if isinstance(other, Bufferable):
            return SumFilterbank((self, other), (-1, 1))
        else:
            func = lambda x: other-x
            return FunctionFilterbank(self, func)

    def __mul__(self, other):
        if isinstance(other, Bufferable):
            func = lambda x, y: x*y
            return FunctionFilterbank((self, other), func)
        else:
            func = lambda x: x*other
            return FunctionFilterbank(self, func)        
    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Bufferable):
            func = lambda x, y: x/y
            return FunctionFilterbank((self, other), func)
        else:
            func = lambda x: x/other
            return FunctionFilterbank(self, func)        

    def __rdiv__(self, other):
        # Note __rdiv__ returns other/self
        if isinstance(other, Bufferable):
            func = lambda x, y: x/y
            return FunctionFilterbank((other, self), func)
        else:
            func = lambda x: other/x
            return FunctionFilterbank(self, func)        


class FunctionFilterbank(Filterbank):
    '''
    Filterbank that just applies a given function. The function should take
    as many arguments as there are sources.
    '''
    def __init__(self, source, func):
        if isinstance(source, Bufferable):
            source = (source,)
        Filterbank.__init__(self, source)
        self.func = func

    def buffer_fetch_next(self, samples):
        start = self.cached_buffer_end
        end = start+samples
        inputs = tuple(s.buffer_fetch(start, end) for s in self.source)
        return self.func(*inputs)


class SumFilterbank(FunctionFilterbank):
    '''  
    Sum filterbanks together with a given weight vectors
    '''
    def __init__(self, source, weights=None):
        if weights is None:
            weights = ones(len(source))
        self.weights = weights
        func = lambda *inputs: sum(input*w for input, w in zip(inputs, weights))
        FunctionFilterbank.__init__(self, source, func)


class DoNothingFilterbank(Filterbank):
    '''
    Filterbank that does nothing to its input.
    
    Useful for removing a set of filters without having to rewrite your code.
    '''
    def buffer_apply(self, input):
        return input

#TODO: InterleaveChannels
#TODO: SerialChannels 
