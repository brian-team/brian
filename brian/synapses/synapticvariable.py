'''
Synaptic variables.
'''
import numpy as np
from brian.log import log_debug
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

    .. automethod:: brian.synapses.synapticvariable.SynapticVariable.to_matrix
    
    '''
    def __init__(self, data, synapses, name):
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
        synapses = self.synapses.synapse_index(i)
        if isinstance(value, str):
            value = self._interpret(value, synapses, level+1)
        self.data[synapses] = value
        
    def _interpret(self, value, synapses, level):
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
        return eval(code, _namespace)
        
    def to_matrix(self, multiple_synapses='last'):
        '''
        Returns the wanted state as a matrix of shape (# presynaptic neurons, # postsynaptic neurons) for visualization purposes. 
        The returned array value at [i,j] is the value of the wanted synaptic variable for the synapse between (i, j). If not synapse exists between those two neurons, then the value is ``np.nan``.

        * Dealing with multiple synapses between two neurons

        Outputting a 2D matrix is not generally possible, because multiple synapses can exist for a given pair or pre- and post-synaptic neurons.
        In this case, the state values for all the synapses between neurons i and j are aggregated in the (i, j) position of the matrix. This is done according to the ``multiple_synapses`` keyword argument which can be changed:
        
        ``mutiple_synapses = 'last'`` (default) takes the last value
        
        ``mutiple_synapses = 'first'`` takes the first value
        
        ``mutiple_synapses = 'min'`` takes the min of the values
        
        ``mutiple_synapses = 'max'`` takes the max of the values
        
        ``mutiple_synapses = 'sum'`` takes the sum of the values
        
        Please note that this function should be used for visualization, and should not be used to store or reload synaptic variable values. 
        If you want to do so, refer to the documentation at :meth:`Synapses.save_connectivity`.
        '''
        
        Nsource = len(self.synapses.source)
        Ntarget = len(self.synapses.target)
        Nsynapses = len(self.synapses)
        
        output = np.ones((Nsource, Ntarget)) * np.nan

        for isyn in xrange(Nsynapses):
            # this is the couple index of the presynaptic and postsynaptic neurons.
            curidx = (self.synapses.presynaptic[isyn], self.synapses.postsynaptic[isyn])
            if multiple_synapses == 'last' or np.isnan(output[curidx]):
                # no previously found synapse, or we want the last one
                output[curidx] = self.data[isyn]
            elif multiple_synapses == 'first':
                # if we want the first one, it's already set
                pass
            else:
                # in this case, we try to use the numpy function as named by the multiple_synapses keyword
                # it's a trick to make this code impossible to understand.
                try: 
                    # there already was a synapse, we aggregate data
                    exec('output[curidx] = np.'+multiple_synapses+'([output[curidx], self.data[isyn]])')
                except AttributeError:
                    log_debug('brian.synapticvariable', 'Couldn\'t figure out how to handle multiple synapses when creating matrix')
                    raise

        return output
                
    

class SynapticDelayVariable(SynapticVariable):
    '''
    A synaptic variable that is a delay.
    The main difference with :class:`~brian.synapses.synapticvariable.SynapticVariable`
    is that delays are stored as integers (timebins) but
    accessed as absolute times (in seconds).
    
    TODO: pass the clock as argument.
    '''
    def __init__(self, data, synapses, name):
        SynapticVariable.__init__(self, data, synapses, name)
        
    def __getitem__(self, i):
        return SynapticVariable.__getitem__(self, i)*self.synapses.clock.dt

    def __setitem__(self, i, value, level=1):
        # will not work with computed values (strings)
        synapses = self.synapses.synapse_index(i)
        if isinstance(value,str):
            value = self._interpret(value,synapses,level+1)
        self.data[synapses] = np.array(np.array(value)/self.synapses.clock.dt,dtype=self.data.dtype)
    
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
