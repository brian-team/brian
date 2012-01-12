'''
Synaptic variables
'''
from brian import * # ugly import
import numpy as np

__all__=['SynapticVariable','SynapticDelayVariable','slice_to_array','neuron_indexes']

class SynapticVariable(object):
    '''
    A synaptic variable is a vector that is returned by Synapses.__getattr__,
    and that can be accessed with getitem with 2 or 3 arguments.
    
    When accessed with one argument (S.w[12]), the argument is considered as
    a synapse index (not a presynaptic neuron index). This is not the same
    behaviour as a sparse array.
    
    TODO:
    * assignment of strings
    '''
    def __init__(self,data,synapses):
        '''
        data: underlying vector
        synapses: Synapses object
        '''
        self.data=data
        self.synapses=synapses
        
    def __getitem__(self,i):
        return self.data[self.synapse_index(i)]

    def __setitem__(self,i,value):
        self.data[self.synapse_index(i)]=value

    def synapse_index(self,i):
        '''
        Returns the synapse index correspond to tuple or slice i
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
            return self.synapse_index(i[:2])[i[2]]
        return i

class SynapticDelayVariable(SynapticVariable):
    def __init__(self,data,synapses):
        SynapticVariable.__init__(self,data,synapses)
        
    def __getitem__(self,i):
        return SynapticVariable.__getitem__(self.i)*self.synapses.clock.dt

    def __setitem__(self,i,value):
        # will not work with computed values (strings)
        SynapticVariable.__setitem__(self,i,array(array(value)/self.synapses.clock.dt,dtype=self.data.dtype))

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
    which can be a integer, an array or a subgroup.
    P is the neuron group.
    '''
    if isinstance(x,NeuronGroup): # it should be checked that x is actually a subgroup of P
        i0=x._origin - P._origin # offset of the subgroup x in P
        return arange(i0,i0+len(x))
    else:
        return slice_to_array(x,N=len(P))      
