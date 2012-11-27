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
from numpy import zeros, ones, isscalar, diag_indices, pi
from numpy.linalg import solve
import numpy
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
        #self.Cm = Cm # could be a vector?
        self.Cm = ones(len(self))*Cm #  Temporary hack - so that it can be a vector, later
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


class SpatialStateUpdater_old(StateUpdater):
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
        ** Note: why are the coefficients not symmetrical then? **
        
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
        # It seems that Aminus(1) = Aminus[0]

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
    
class SpatialStateUpdater_new(StateUpdater):
    """
    State updater for compartmental models, with several branches.
    """
    def __init__(self, neuron, clock=None):
        self.eqs = neuron._eqs
        self.neuron = neuron
        self._isprepared = False
        self._state_updater=neuron._state_updater # to update the currents

    def cut_branches(self,morphology):
        '''
        Recursively cut the branches by setting zero axial resistances.
        '''
        self.invr[morphology._origin]=0
        for kid in (morphology.children):
            self.cut_branches(kid)
    
    def number_branches(self,morphology,n=0,parent=-1):
        '''
        Recursively number the branches and return their total number.
        n is the index number of the current branch.
        parent is the index number of the parent branch.
        '''
        morphology.index=n
        morphology.parent=parent
        nbranches=1
        for kid in (morphology.children):
            nbranches+=self.number_branches(kid,n+nbranches,n)
        return nbranches
    
    def prepare(self):
        '''
        Preparation of data structures.
        See the relevant document.
        '''
        # Correction for soma (a bit of a hack), so that it has negligible axial resistance
        if self.neuron.morphology.type=='soma':
            self.neuron.length[0]=self.neuron.diameter[0]*0.01
        # Inverse axial resistance
        self.invr=zeros(len(self.neuron))
        self.invr[1:]=pi/(2*self.neuron.Ri)*(self.neuron.diameter[:-1]*self.neuron.diameter[1:])/\
                   (self.neuron.length[:-1]+self.neuron.length[1:])
        # Note: this would give nan for the soma
        self.cut_branches(self.neuron.morphology)
        
        # Linear systems
        # The particular solution
        '''a[i,j]=ab[u+i-j,j]''' # u is the number of upper diagonals = 1
        self.ab_star=zeros((3,len(self.neuron)))
        self.ab_star[0,1:]=self.invr[1:]/self.neuron.area[:-1]
        self.ab_star[2,:-1]=self.invr[1:]/self.neuron.area[1:]
        self.ab_star[1,:]=-self.neuron.Cm/self.neuron.clock.dt-self.invr/self.neuron.area
        self.ab_star[1,:-1]-=self.invr[1:]/self.neuron.area[:-1]
        # Homogeneous solutions
        self.ab_plus=zeros((3,len(self.neuron)))
        self.ab_minus=zeros((3,len(self.neuron)))
        self.ab_plus[:]=self.ab_star
        self.ab_minus[:]=self.ab_star
        self.b_plus=zeros(len(self.neuron))
        self.b_minus=zeros(len(self.neuron))
        # Solutions
        self.v_star=zeros(len(self.neuron))
        self.u_plus=zeros(len(self.neuron))
        self.u_minus=zeros(len(self.neuron))
        
        # Boundary conditions
        self.boundary_conditions(self.neuron.morphology)
        
        # Linear system for connecting branches
        n=1+self.number_branches(self.neuron.morphology) # number of nodes (2 for the root)
        self.P=zeros((n,n)) # matrix
        self.B=zeros(n) # vector RHS
        self.V=zeros(n) # solution = voltages at nodes

    def boundary_conditions(self,morphology):
        '''
        Recursively sets the boundary conditions in the linear systems.
        '''
        first=morphology._origin # first compartment
        last=first+len(morphology.x)-1 # last compartment
        # Inverse axial resistances at the ends: r0 and rn
        morphology.invr0=float(pi/(2*self.neuron.Ri)*self.neuron.diameter[first]**2/self.neuron.length[first])
        morphology.invrn=float(pi/(2*self.neuron.Ri)*self.neuron.diameter[last]**2/self.neuron.length[last])
        # Correction for boundary conditions
        self.ab_star[1,first]-=float(morphology.invr0/self.neuron.area[first]) # because of units problems
        self.ab_star[1,last]-=float(morphology.invrn/self.neuron.area[last])
        self.ab_plus[1,first]-=float(morphology.invr0/self.neuron.area[first]) # because of units problems
        self.ab_plus[1,last]-=float(morphology.invrn/self.neuron.area[last])
        self.ab_minus[1,first]-=float(morphology.invr0/self.neuron.area[first]) # because of units problems
        self.ab_minus[1,last]-=float(morphology.invrn/self.neuron.area[last])
        # RHS for homogeneous solutions
        self.b_plus[last]=-float(morphology.invrn/self.neuron.area[last])
        self.b_minus[first]=-float(morphology.invr0/self.neuron.area[first])
        # Recursive call
        for kid in (morphology.children):
            self.boundary_conditions(kid)

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
        
        That is:
        a[i,j]=ab[1+i-j,j]
        ab[1,:]=main diagonal
        ab[0,1:]=upper diagonal
        ab[2,:-1]=lower diagonal        '''
        # Particular solution
        b=-neuron.Cm/neuron.clock.dt*neuron.v-neuron._I0
        ab = zeros((3,len(neuron)))
        ab[:]=self.ab_star
        ab[1,:]-=neuron._gtot
        self.v_star[:]=solve_banded((1,1),ab,b,overwrite_ab=True,overwrite_b=True)
        # Homogeneous solutions
        b[:]=self.b_plus
        ab[:]=self.ab_plus 
        ab[1,:]-=neuron._gtot
        self.u_plus[:]=solve_banded((1,1),ab,b,overwrite_ab=True,overwrite_b=True)
        b[:]=self.b_minus
        ab[:]=self.ab_minus 
        ab[1,:]-=neuron._gtot
        self.u_minus[:]=solve_banded((1,1),ab,b,overwrite_ab=True,overwrite_b=True)
        # Solve the linear system connecting branches
        self.P[:]=0
        self.B[:]=0
        self.fill_matrix(self.neuron.morphology)
        self.V = solve(self.P,self.B)
        # Calculate solutions by linear combination
        self.linear_combination(self.neuron.morphology)
        
    def linear_combination(self,morphology):
        '''
        Calculates solutions by linear combination
        '''
        first=morphology._origin # first compartment
        last=first+len(morphology.x)-1 # last compartment
        i=morphology.index+1
        i_parent=morphology.parent+1
        self.neuron.v[first:last+1]=self.v_star[first:last+1]+self.V[i_parent]*self.u_minus[first:last+1]\
                                                             +self.V[i]*self.u_plus[first:last+1]
        # Recursive call
        for kid in (morphology.children):
            self.linear_combination(kid)

    def fill_matrix(self,morphology):
        '''
        Recursively fills the matrix of the linear system that connects branches together.
        '''
        first=morphology._origin # first compartment
        last=first+len(morphology.x)-1 # last compartment
        i=morphology.index+1
        i_parent=morphology.parent+1
        # Towards parent
        if i==1: # first branch, sealed end
            self.P[0,0]=self.u_minus[first]-1
            self.P[0,1]=self.u_plus[first]
            self.B[0]=-self.v_star[first]
        else:
            self.P[i_parent,i_parent]+=(1-self.u_minus[first])*morphology.invr0
            self.P[i_parent,i]-=self.u_plus[first]*morphology.invr0
            self.B[i_parent]+=self.v_star[first]*morphology.invr0
        # Towards children
        self.P[i,i]=(1-self.u_plus[last])*morphology.invrn
        self.P[i,i_parent]=-self.u_minus[last]*morphology.invrn
        self.B[i]=self.v_star[last]*morphology.invrn
        # Recursive call
        for kid in (morphology.children):
            self.fill_matrix(kid)

    def __len__(self):
        '''
        Number of state variables
        '''
        return len(self.eqs)
    
SpatialStateUpdater = SpatialStateUpdater_new

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
