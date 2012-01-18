'''
Synaptic variables
'''
from brian import * # ugly import
from brian.inspection import *
import numpy as np

__all__=['SynapticVariable','SynapticDelayVariable','slice_to_array','neuron_indexes']

class SynapticVariable(object):
    '''
    A vector of synaptic variables that is returned by Synapses.__getattr__,
    and that can be subscripted with 2 or 3 arguments.
    
    Example usages, where ``S'' is Synapses object:
    
    ``S.w[12]''
        Value of variable w for synapse 12. 
    ``S.w[1,3]''
        Value of variable w for synapses from neuron 1 to neuron 3. This is an array,
        as there can be several synapses for a given neuron pair (e.g. with different
        delays)
    ``S.w[1,3,4]''
        Value of variable w for synapse 4 from neuron 1 to neuron 3.
        
    Indexes can be integers, slices, arrays or groups.
    
    Synaptic variables can be assigned values as follows:
    
    ``S.w[P,Q]=x''
        where x is a float or a 1D array. The number of elements in the array must
        equal the number of selected synapses.
    ``S.w[P,Q]=s''
        where s is a string. The string is Python code that is executed in a single
        vectorised operation, where ``i'' is the presynaptic neuron index (a vector
        of length the number of synapses), ``j'' is the postsynaptic neuron index and
        ``n'' is the number of synapses. The methods ``rand'' and ``randn'' return
        arrays of n random values.
        
    Initialised with arguments:

    ``data''
        Vector of values.
    ``synapses''
        The Synapses object.
    '''
    def __init__(self,data,synapses):
        self.data=data
        self.synapses=synapses
        class Replacer(object): # vectorisation in strings
            def __init__(self, func, n):
                self.n = n
                self.func = func
            def __call__(self):
                return self.func(self.n)
        self._Replacer = Replacer
        
    def __getitem__(self,i):
        return self.data[self.synapse_index(i)]

    def __setitem__(self,i,value,level=1):
        synapses=self.synapse_index(i)
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

    def synapse_index(self,i):
        '''
        Returns the synapse index correspond to i, which can be a tuple or a slice.
        If i is a tuple (m,n), m and n can be an integer, an array, a slice or a subgroup.
        '''
        if not isinstance(i,tuple): # we assume it is directly a synapse index
            return i
        if len(i)==2:
            i,j=i
            i=neuron_indexes(i,self.synapses.source)
            j=neuron_indexes(j,self.synapses.target)
            synapsetype=self.synapses.synapses_pre[0].dtype
            synapses_pre=array(hstack([self.synapses.synapses_pre[k] for k in i]),dtype=synapsetype)
            synapses_post=array(hstack([self.synapses.synapses_post[k] for k in j]),dtype=synapsetype)
            return np.intersect1d(synapses_pre, synapses_post,assume_unique=True)
        elif len(i)==3: # 3rd coordinate is synapse number
            if i[0] is scalar and i[1] is scalar:
                return self.synapse_index(i[:2])[i[2]]
            else:
                raise NotImplementedError,"The first two coordinates must be integers"
        return i

class SynapticDelayVariable(SynapticVariable):
    '''
    A synaptic variable that is a delay.
    The main difference with SynapticVariable is that
    delays are stored as integers (timebins) but
    accessed as absolute times (in seconds).
    
    TODO: pass the clock as argument.
    '''
    def __init__(self,data,synapses):
        SynapticVariable.__init__(self,data,synapses)
        
    def __getitem__(self,i):
        return SynapticVariable.__getitem__(self,i)*self.synapses.clock.dt

    def __setitem__(self,i,value,level=1):
        # will not work with computed values (strings)
        synapses=self.synapse_index(i)
        if isinstance(value,str):
            value=self._interpret(value,synapses,level+1)
        self.data[synapses]=array(array(value)/self.synapses.clock.dt,dtype=self.data.dtype)

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
        return arange(start,stop,step)
    elif isscalar(s): # if not a slice (e.g. an int) then we return it as an array of a single element
        return array([s])
    else: # array or sequence
        return array(s)

def neuron_indexes(x,P):
    '''
    Returns the array of neuron indexes corresponding to x,
    which can be a integer, an array, a slice or a subgroup.
    P is the neuron group.
    '''
    if isinstance(x,NeuronGroup): # it should be checked that x is actually a subgroup of P
        i0=x._origin - P._origin # offset of the subgroup x in P
        return arange(i0,i0+len(x))
    else:
        return slice_to_array(x,N=len(P))      
