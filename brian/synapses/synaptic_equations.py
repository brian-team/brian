"""
Synaptic equations
"""
import re

from brian.equations import Equations
from brian.units import Quantity, scalar_representation, second
from brian.stdunits import ms

__all__=['SynapticEquations']

class SynapticEquations(Equations):
    '''
    Equations for the Synapses class.
    The only difference with :class:`Equations` is that differential equations
    can be marked for an event-driven implementation, e.g.:
    
    ``dx/dt=-x/tau : 1 (event-driven)``
    '''
    def __init__(self, expr='', level=0, **kwds):
        self._eventdriven={} # dictionary event driven variables (RHS)
        Equations.__init__(self, expr, level=level+1, **kwds)

    def add_diffeq(self, name, eq, unit, global_namespace={}, local_namespace={}, nonzero=True):
        '''
        unit may contain "(event-driven)" 
        '''
        pattern=re.compile("(\w+)\s*\(event\-driven\)\s*")
        result=pattern.match(unit) # event-driven
        if result:
            unit, = result.groups()
            # We treat as a diff eq to get the correct namespaces
            Equations.add_diffeq(self,name, eq, unit, global_namespace, local_namespace, nonzero=False)
            if isinstance(unit, Quantity):
                unit = scalar_representation(unit)
            self._string[name]='0*' + unit + '/second'
            self._namespace[name]['second']=second
            # then add it to the list of event-driven variables
            self._eventdriven[name]=eq
        else:
            Equations.add_diffeq(self,name, eq, unit, global_namespace, local_namespace, nonzero)

if __name__=='__main__':
    tau_pre=10*ms
    tau_post=20*ms
    eqs='''w:1
           dx/dt=-x/tau_pre : 1 
           dA_pre/dt=-A_pre/tau_pre : 1 (event-driven)
           dA_post/dt=-A_post/tau_post : 1 (event-driven)'''
    eqs=SynapticEquations(eqs)
    print eqs
    print eqs._eventdriven
