'''
Synaptic variables
'''
from brian import * # ugly import
import numpy as np

__all__=['SynapticVariable','SynapticDelayVariable']

class SynapticVariable(object):
    '''
    A synaptic variable is a vector that is returned by Synapses.__getattr__,
    and that can be accessed with getitem with 2 or 3 arguments.
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
        if not isinstance(i,tuple):
            raise NotImplementedError,'Subscripting with a single index is not implemented yet'
        if len(i)==2:
            i,j=i
            if isscalar(i) and isscalar(j):
                synapses_pre=self.synapses.synapses_pre[i]
                synapses_post=self.synapses.synapses_post[j]
                return np.intersect1d(synapses_pre, synapses_post)
            else:
                raise NotImplementedError,'Not implemented yet'
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
