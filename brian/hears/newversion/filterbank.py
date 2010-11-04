from brian import *
from scipy import signal, weave, random
from bufferable import Bufferable

__all__ = ['Filterbank',
           'ChainedFilterbank',
           'FunctionFilterbank',
           'MixFilterbank',
           ]

class Filterbank(Bufferable):
    '''
    Generalised filterbank object
    
    This class is a base class not designed to be instantiated. A Filterbank
    object should define the interface of :class:`Bufferable`, as well as
    defining a ``source`` attribute giving the source :class:`Bufferable`
    object.
    
    The class in addition defines some operator methods for adding, multiplying,
    etc. of filterbanks.
    '''
    
    source = NotImplemented
    
    def __add__ (fb1, fb2):
        return MixFilterbank(fb1, fb2)
        
    def __sub__ (fb1, fb2):
        return MixFilterbank(fb1, fb2, array([1, -1]))  
        
    def __rmul__(fb, scalar):
        func = lambda x: scalar*x
        return FilterbankChain([fb, FunctionFilterbank(fb.fs, fb.N, func)])

### TODO: Update to work with new interface
class ChainedFilterbank(Filterbank):
    '''
    Chains multiple filterbanks together
    
    Usage::
    
        ChainedFilterbank(filterbanks)
        
    Where ``filterbanks`` is a list of filters. The signal is fed into the first in
    this list, and then the output of that is fed into the next, and so on.
    '''
    def __init__(self, filterbanks):
        self.filterbanks=filterbanks
        self.fs=filterbanks[0].fs
        self.N=filterbanks[0].N
        
    def timestep(self, input):
        for fb in self.filterbanks:
            input=fb.timestep(input)
        return input

    def __len__(self):
        return len(self.filterbanks[0])

    samplerate=property(fget=lambda self:self.filterbanks[0].samplerate)

# TODO: update all of this with the new interface/buffering mechanism
class FunctionFilterbank(Filterbank):
    '''
    Filterbank that just applies a given function
    '''
    def __init__(self, samplerate, N, func):
        self.fs=samplerate
        self.N=N
        self.func=func

    def timestep(self, input):
        # self.output[:]=self.func(input)
        return self.func(input)

    def __len__(self):
        return self.N

    samplerate=property(fget=lambda self:self.fs)

# TODO: update all of this with the new interface/buffering mechanism
class MixFilterbank(Filterbank):
    '''  
    Mix filterbanks together with a given weight vectors
    '''
    
    def __init__(self, *args, **kwargs):#weights=None,

        if kwargs.get('weights')==None:
            weights=ones((len(args)))
        else:
            weights=kwargs.get('weights')
        self.filterbanks=args
        #print  args
        self.fs=args[0].fs
        self.nbr_fb=len(self.filterbanks)
        self.N=args[0].N
        self.output=zeros(self.N)
        self.weights=weights

    def timestep(self, input):
        self.output=zeros(self.N)
        for ind, fb in zip(range((self.nbr_fb)), self.filterbanks):
            self.output+=self.weights[ind]*fb.timestep(input)
        return self.output

    def __len__(self):
        return self.N
    samplerate=property(fget=lambda self:self.fs)
