'''
Distributed simulations on clusters, using MPI.

Groups hosted on other machines are mapped to virtual groups.
Connection objects do not change (simply, source or target may be virtual).
The ClientNetwork object runs the network normally, except it sends and receives spikes
from times to times to other machines.
The ServerNetwork object creates ClientNetwork objects and distributes them over the
machines. It does not need supervise the simulation.

How about creation? (in particular, connection objects should be created on the target machine)
    It should be possible to create the objects on the target machines, e.g.:
    * create the group/connection on the server
    * communicate the object to the client
    * convert to virtual group on the server

How about monitoring?
    Simply communicate the monitors to the machine hosting the relevant neuron group.

How about plasticity?
'''
from brian.neurongroup import *
from brian.network import *

class VirtualGroup(NeuronGroup):
    '''
    A group that is hosted on another machine.
    A number of methods will raise an exception (e.g. accessing the state variables).
    '''
    def __init__(self,group,machine):
        '''
        Initializes the virtual group:
            group: real group
            machine: machine id
        '''
        # How do we deal with clocks?
        self._S0=group._S0
        self.staticvars=group.staticvars
        self.var_index=group.var_index
        self._max_delay=group._max_delay
        self.LS=group.LS
        self._owner=machine
        self._length=len(group)
        self._numstates=group.num_states()        
 
    # Update and reset are disabled
    def update(self):
        pass    
    def reset(self):
        pass
    def __len__(self):
        return self._length
    def num_states(self):
        return self._numstates    
    def __repr__(self):
        return 'Virtual group of '+str(len(self))+' neurons'
    
class ServerNetwork(Network):
    '''
    Network class for running a simulation over a cluster.
    The server manages the clients.
    '''
    pass

class ClientNetwork(Network):
    '''
    Network class for running a simulation over a cluster, client side.
    '''
    pass
