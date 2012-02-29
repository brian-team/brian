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
from brian.clock import guess_clock
from itertools import count
from brian.neurongroup import NeuronGroup
from scipy.linalg import solve_banded
from numpy import zeros
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
                 compile=False, freeze=False, Cm=0.9 * uF / cm ** 2, Ri=150 * ohm * cm):
        clock = guess_clock(clock)
        N = len(morphology) # number of compartments

        if isinstance(model, str):
            model = Equations(model, level=level + 1)

        model += Equations('''
        v:volt # membrane potential
        ''')

        # Process model equations (Im) to extract total conductance and the remaining current
        if use_sympy:
            try:
                membrane_eq=model._string['Im'] # the membrane equation
            except:
                raise TypeError,"The transmembrane current Im must be defined"
            # Check conditional linearity
            ids=get_identifiers(membrane_eq)
            _namespace=dict.fromkeys(ids,1.) # there is a possibility of problems here (division by zero)
            _namespace['v']=AffineFunction()
            eval(membrane_eq,model._namespace['v'],_namespace)
            try:
                eval(membrane_eq,model._namespace['v'],_namespace)
            except: # not linear
                raise TypeError,"The membrane current must be linear with respect to v"
            # Extracts the total conductance from Im, and the remaining current
            z=symbolic_eval(membrane_eq)
            symbol_v=sympy.Symbol('v')
            b=z.subs(symbol_v,0)
            a=-sympy.simplify(z.subs(symbol_v,1)-b)
            gtot_str="_gtot="+str(a)+": siemens/cm**2"
            I0_str="_I0="+str(b)+": amp/cm**2"
            model+=Equations(gtot_str+"\n"+I0_str,level=level+1) # better: explicit insertion with namespace of v
        else:
            raise TypeError,"The Sympy package must be installed for SpatialNeuron"

        # Equations for morphology (isn't it a duplicate??)
        eqs_morphology = Equations("""
        diameter : um
        length : um
        x : um
        y : um
        z : um
        area : um**2
        """)

        full_model = model + eqs_morphology

        # Create the state updater
        NeuronGroup.__init__(self, N, model=full_model, threshold=threshold, reset=reset, refractory=refractory,
                             level=level + 1, clock=clock, unit_checking=unit_checking, implicit=True)

        #Group.__init__(self, full_model, N, unit_checking=unit_checking, level=level+1)
        #self._eqs = model
        #var_names = full_model._diffeq_names
        self.Cm = Cm # could be a vector?
        self.Ri = Ri
        self._state_updater = SpatialStateUpdater(self, clock)
        #S0 = {}
        # Fill missing units
        #for key, value in full_model._units.iteritems():
        #    if not key in S0:
        #        S0[key] = 0 * value
        #self._S0 = [0] * len(var_names)
        #for var, i in zip(var_names, count()):
        #    self._S0[i] = S0[var]

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
    
    For the moment I assume there is a single branch (=axon).
    """
    def __init__(self, neuron, clock=None):
        self.eqs = neuron._eqs
        self.neuron = neuron
        self._isprepared = False
        self._state_updater=neuron._state_updater # to update the currents

    def prepare(self):
        '''
        From Hines 1984 paper, discrete formula is:
        A_plus*V(i+1)-(A_plus+A_minus)*V(i)+A_minus*V(i-1)=Cm/dt*(V(i,t+dt)-V(i,t))+gtot(i)*V(i)-I0(i)
       
        A_plus: i->i+1
        A_minus: i->i-1
        
        This gives the following tridiagonal system:
        A_plus*V(i+1)-(Cm/dt+gtot(i)+A_plus+A_minus)*V(i)+A_minus*V(i-1)=-Cm/dt*V(i,t)-I0(i)
        
        Boundaries, one simple possibility (sealed ends):
        -(Cm/dt+gtot(n)+A_minus)*V(n)+A_minus*V(n-1)=-Cm/dt*V(n,t)-I0(n)
        A_plus*V(1)-(Cm/dt+gtot(0)+A_plus)*V(0)=-Cm/dt*V(0,t)-I0(0)
        
        For the domain decomposition idea, most can be precalculated, and only the diagonal
        elements (gtot(i)) and the RHS will change.
        '''
        mid_diameter=.5*(self.neuron.diameter[:-1]+self.neuron.diameter[1:]) # i -> i+1
        self.Aplus=mid_diameter**2/(4*self.neuron.diameter[:-1]*self.neuron.length[:-1]**2*self.neuron.Ri)
        self.Aminus=mid_diameter**2/(4*self.neuron.diameter[1:]*self.neuron.length[1:]**2*self.neuron.Ri)

    def __call__(self, neuron):
        '''
        Updates the state variables.
        '''
        if not self._isprepared:
            self.prepare()
            self._isprepared=True
        # Update the membrane currents (should it be after (no real difference though))
        self._state_updater(neuron)
        '''
        x=solve_banded((lower,upper),ab,b)
        lower = number of lower diagonals = 1
        upper = number of upper diagonals = 1
        ab = array(l+u+1,M)
            each row is one diagonal
        a[i,j]=ab[u+i-j,j]
        '''
        b=-neuron.Cm/neuron.clock.dt*neuron.v-neuron._I0
        ab = zeros((3,len(neuron))) # part of it could be precomputed
        # a[i,j]=ab[1+i-j,j]
        # ab[1,:]=main diagonal
        # ab[0,1:]=upper diagonal
        # ab[2,:-1]=lower diagonal
        ab[0,1:]=self.Aplus
        ab[2,:-1]=self.Aminus
        ab[1,:]=-neuron.Cm/neuron.clock.dt-neuron._gtot
        ab[1,1:]-=self.Aminus
        ab[1,:-1]-=self.Aplus
        neuron.v=solve_banded((1,1),ab,b,overwrite_ab=True,overwrite_b=True)

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
    neuron = SpatialNeuron(morphology=morpho, threshold="axon[50*um].v>0*mV", model=eqs, refractory=4 * ms, Cm=0.9 * uF / cm ** 2, Ri=150 * ohm * cm)
    neuron.axon[0 * um:50 * um].gl = 1e-3 * siemens / cm ** 2
    print sum(neuron.axon.gl)
    print neuron.axon[40 * um].gl
    #branch=neuron.axon[0*um:50*um]
    neuron.morphology.plot()
    show()
