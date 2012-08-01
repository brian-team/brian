'''
Synaptic variables.
'''
import numpy as np

from brian.inspection import namespace

__all__=['SynapticVariable','SynapticDelayVariable','slice_to_array']

class SynapticVariable(object):
    '''
    A vector of synaptic variables that is returned by :meth:`Synapses.__getattr__`,
    and that can be subscripted with 2 or 3 arguments.
    
    Example usages, where ``S`` is Synapses object:
    
    ``S.w[12]``
        Value of variable w for synapse 12. 
    ``S.w[1,3]``
        Value of variable w for synapses from neuron 1 to neuron 3. This is an array,
        as there can be several synapses for a given neuron pair (e.g. with different
        delays)
    ``S.w[1,3,4]``
        Value of variable w for synapse 4 from neuron 1 to neuron 3.
        
    Indexes can be integers, slices, arrays or groups.
    
    Synaptic variables can be assigned values as follows:
    
    ``S.w[P,Q]=x``
        where x is a float or a 1D array. The number of elements in the array must
        equal the number of selected synapses.
    ``S.w[P,Q]=s``
        where s is a string. The string is Python code that is executed in a single
        vectorised operation, where ``i`` is the presynaptic neuron index (a vector
        of length the number of synapses), ``j`` is the postsynaptic neuron index and
        ``n`` is the number of synapses. The methods ``rand`` and ``randn`` return
        arrays of n random values.
        
    Initialised with arguments:

    ``data``
        Vector of values.
    ``synapses``
        The Synapses object.
    '''
    def __init__(self,data,synapses,name):
        self.data=data
        self.name=name
        self.synapses=synapses
        class Replacer(object): # vectorisation in strings
            def __init__(self, func, n):
                self.n = n
                self.func = func
            def __call__(self):
                return self.func(self.n)
        self._Replacer = Replacer
        
    def __getitem__(self,i):
        return self.data[self.synapses.synapse_index(i)]

    def __setitem__(self,i,value,level=1):
        synapses=self.synapses.synapse_index(i)
        if isinstance(value,str):
            value=self._interpret(value,synapses,level+1)
        self.data[synapses]=value
        
    def _interpret(self,value,synapses,level):
        '''
        Interprets value string in the context of the synaptic indexes synapses
        '''
        _namespace = namespace(value, level=level)
        code = compile(value, "StringAssignment", "eval")
        synapses=slice_to_array(synapses,N=len(self.synapses))
        _namespace['n']=len(synapses)
        _namespace['i']=self.synapses.presynaptic[synapses]
        _namespace['j']=self.synapses.postsynaptic[synapses]
        for var in self.synapses.var_index: # maybe synaptic variables should have higher priority
            if isinstance(var,str):
                _namespace[var] = self.synapses.state(var)[synapses]
        _namespace['rand'] = self._Replacer(np.random.rand, len(synapses))
        _namespace['randn'] = self._Replacer(np.random.randn, len(synapses))
        return eval(code,_namespace)

class SynapticDelayVariable(SynapticVariable):
    '''
    A synaptic variable that is a delay.
    The main difference with :class:`~brian.synapses.synapticvariable.SynapticVariable`
    is that delays are stored as integers (timebins) but
    accessed as absolute times (in seconds).
    
    TODO: pass the clock as argument.
    '''
    def __init__(self,data,synapses,name):
        SynapticVariable.__init__(self,data,synapses,name)
        
    def __getitem__(self,i):
        return SynapticVariable.__getitem__(self,i)*self.synapses.clock.dt

    def __setitem__(self,i,value,level=1):
        # will not work with computed values (strings)
        synapses=self.synapses.synapse_index(i)
        if isinstance(value,str):
            value=self._interpret(value,synapses,level+1)
        self.data[synapses]=np.array(np.array(value)/self.synapses.clock.dt,dtype=self.data.dtype)

def slice_to_array(s,N=None):
    '''
    Converts a slice s, single int or array to the corresponding array of integers.
    N is the maximum number of elements, this is used to handle negative numbers
    in the slice.
    '''
    if isinstance(s,slice):
        start=s.start or 0
        stop=s.stop or N
        step=s.step
        if stop<0 and N is not None:
            stop=N+stop
        return np.arange(start,stop,step)
    elif np.isscalar(s): # if not a slice (e.g. an int) then we return it as an array of a single element
        return np.array([s])
    else: # array or sequence
        return np.array(s)
