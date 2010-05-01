'''
Compartmental neurons
See BEP-15
'''
from morphology import *
from brian.stdunits import *
from brian.units import *
from brian.reset import NoReset
from brian.stateupdater import StateUpdater
from brian.equations import Equations
from brian.group import Group
from itertools import count
from brian.neurongroup import NeuronGroup

__all__ = ['SpatialNeuron','CompartmentalNeuron']

class SpatialNeuron(NeuronGroup):
    """
    Compartmental model with morphology.
    """
    def __init__(self, morphology=None, model=None, threshold=None, reset=NoReset(),
                 refractory=0*ms, level=0,
                 clock=None, unit_checking=True,
                 compile=False, freeze=False,cm=0.9*uF/cm**2,Ri=150*ohm*cm):
        clock=guess_clock(clock)
        N=len(morphology) # number of compartments
        
        # Equations for morphology
        eqs_morphology=Equations("""
        diameter : um
        length : um
        x : um
        y : um
        z : um
        area : um**2
        """)
        
        # Create the state updater
        if isinstance(model,str):
            model = Equations(model, level=level+1)
        model+=Equations('v:volt') # membrane potential
        full_model=model+eqs_morphology
        self._eqs = model
        self._state_updater = SpatialStateUpdater(model,clock)
        var_names = full_model._diffeq_names
        Group.__init__(self, full_model, N, unit_checking=unit_checking)
        S0 = {}
        # Fill missing units
        for key, value in full_model._units.iteritems():
            if not key in S0:
                S0[key] = 0*value
        self._S0 = [0]*len(var_names)
        for var, i in zip(var_names, count()):
            self._S0[i] = S0[var]
        
        NeuronGroup.__init__(self,N,model=self._state_updater,threshold=threshold,reset=reset,refractory=refractory,
                             level=level+1,clock=clock,unit_checking=unit_checking)

        # Insert morphology (TODO)

class SpatialStateUpdater(StateUpdater):
    """
    State updater for compartmental models.
    """
    def __init__(self,eqs,clock=None):
        self.eqs=eqs
    
    def __len__(self):
        '''
        Number of state variables
        '''
        return len(self.eqs)
    
CompartmentalNeuron=SpatialNeuron

if __name__=='__main__':
    from brian import *
    morpho=Morphology('mp_ma_40984_gc2.CNG.swc') # visual L3 pyramidal cell
    print len(morpho),"compartments"
    El=-70*mV
    eqs=''' # The same equations for the whole neuron, but possibly different parameter values
    Im=gl*(El-v) : amp/cm**2 # distributed transmembrane current
    gl : siemens/cm**2 # spatially distributed conductance
    '''
    neuron=SpatialNeuron(morphology=morpho,threshold="axon[50*um].v>0*mV",model=eqs,refractory=4*ms,cm=0.9*uF/cm**2,Ri=150*ohm*cm)
