'''
Compartmental neurons
See BEP-15

TODO:
* Threshold and reset are special (not as normal NeuronGroup because only 1 spike)
* Hines method
* Point processes
* StateMonitor
* neuron.plot('gl')
* Iteration (over the branch or the entire tree?)
'''
from morphology import *
from brian.stdunits import *
from brian.units import *
from brian.reset import NoReset
from brian.stateupdater import StateUpdater
from brian.equations import Equations
from brian.inspection import *
from brian.optimiser import *
from brian.group import Group
from itertools import count
from brian.neurongroup import NeuronGroup
try:
    import sympy
    use_sympy = True
except:
    warnings.warn('sympy not installed: some features in SpatialNeuron will not be available')
    use_sympy = False

__all__ = ['SpatialNeuron', 'CompartmentalNeuron']


class SpatialNeuron(NeuronGroup):
    """
    Compartmental model with morphology.
    """
    def __init__(self, morphology=None, model=None, threshold=None, reset=NoReset(),
                 refractory=0 * ms, level=0,
                 clock=None, unit_checking=True,
                 compile=False, freeze=False, cm=0.9 * uF / cm ** 2, Ri=150 * ohm * cm):
        clock = guess_clock(clock)
        N = len(morphology) # number of compartments

        # Equations for morphology
        eqs_morphology = Equations("""
        diameter : um
        length : um
        x : um
        y : um
        z : um
        area : um**2
        """)

        # Create the state updater
        if isinstance(model, str):
            model = Equations(model, level=level + 1)
        model += Equations('''
        v:volt # membrane potential
        #Im:amp/cm**2 # membrane current (should we have it?)
        ''')
        full_model = model + eqs_morphology
        Group.__init__(self, full_model, N, unit_checking=unit_checking)
        self._eqs = model
        self._state_updater = SpatialStateUpdater(model, clock)
        var_names = full_model._diffeq_names
        self.cm = cm # could be a vector?
        self.Ri = Ri
        S0 = {}
        # Fill missing units
        for key, value in full_model._units.iteritems():
            if not key in S0:
                S0[key] = 0 * value
        self._S0 = [0] * len(var_names)
        for var, i in zip(var_names, count()):
            self._S0[i] = S0[var]

        NeuronGroup.__init__(self, N, model=self._state_updater, threshold=threshold, reset=reset, refractory=refractory,
                             level=level + 1, clock=clock, unit_checking=unit_checking)

        # Insert morphology
        self.morphology = morphology
        self.morphology.compress(diameter=self.diameter, length=self.length, x=self.x, y=self.y, z=self.z, area=self.area)

    def subgroup(self, N): # Subgrouping cannot be done in this way
        raise NotImplementedError

    def __getitem__(self, x):
        '''
        Subgrouping mechanism.
        self['axon'] returns the subtree named "axon".
        
        TODO:
        self[:] returns the full branch.
        '''
        morpho = self.morphology[x]
        N = self[morpho._origin:morpho._origin + len(morpho)]
        N.morphology = morpho
        return N

    def __getattr__(self, x):
        if (x != 'morphology') and ((x in self.morphology._namedkid) or all([c in 'LR123456789' for c in x])): # subtree
            return self[x]
        else:
            return NeuronGroup.__getattr__(self, x)


class SpatialStateUpdater(StateUpdater):
    """
    State updater for compartmental models.
    """
    def __init__(self, eqs, clock=None):
        self.eqs = eqs
        if use_sympy:
            '''
            TODO:
            * Move this to SpatialNeuron (equation crunching)
            '''
            membrane_eq=self.eqs._string['Im'] # the membrane equation
            # Check conditional linearity
            ids=get_identifiers(membrane_eq)
            _namespace=dict.fromkeys(ids,1.) # there is a possibility of problems here (division by zero)
            _namespace['v']=AffineFunction()
            try:
                eval(membrane_eq,eqs._namespace['v'],_namespace)
            except: # not linear
                raise TypeError,"The membrane current must be linear with respect to v"
            '''
            This extracts the total conductance from Im, and the
            remaining current.
            '''
            z=symbolic_eval(membrane_eq)
            symbol_v=sympy.Symbol('v')
            b=z.subs(symbol_v,0)
            a=-sympy.simplify(z.subs(symbol_v,1)-b)
            gtot_str="_gtot="+str(a)+": siemens/cm**2"
            I0_str="_I0="+str(b)+": amp/cm**2"
            """
            TODO Here: add this in the equations object, and possibly replace
            or delete Im.
            """
            print gtot_str
            print I0_str

    def __len__(self):
        '''
        Number of state variables
        '''
        return len(self.eqs)

CompartmentalNeuron = SpatialNeuron

if __name__ == '__main__':
    from brian import *
    morpho = Morphology('oi24rpy1.CNG.swc') # visual L3 pyramidal cell
    print len(morpho), "compartments"
    El = -70 * mV
    ENa = 50 * mV
    eqs = ''' # The same equations for the whole neuron, but possibly different parameter values
    Im=gl*(El-v)+gNa*m**3*(ENa-v) : amp/cm**2 # distributed transmembrane current
    gl : siemens/cm**2 # spatially distributed conductance
    gNa : siemens/cm**2
    dm/dt=(minf-m)/(0.3*ms) : 1
    minf=1./(1+exp(-(v+30*mV)/(6*mV))) : 1
    '''
    neuron = SpatialNeuron(morphology=morpho, threshold="axon[50*um].v>0*mV", model=eqs, refractory=4 * ms, cm=0.9 * uF / cm ** 2, Ri=150 * ohm * cm)
    neuron.axon[0 * um:50 * um].gl = 1e-3 * siemens / cm ** 2
    print sum(neuron.axon.gl)
    print neuron.axon[40 * um].gl
    #branch=neuron.axon[0*um:50*um]
    neuron.morphology.plot()
    show()
