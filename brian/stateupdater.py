# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
'''
Neuron StateUpdaters
'''

__all__ = ['StateUpdater', 'LinearStateUpdater', 'NonlinearStateUpdater',
           'SynapticNoise', 'LazyStateUpdater', 'magic_state_updater',
           'FunStateUpdater', 'get_linear_equations']

#from scipy.weave import blitz
from numpy import *
from scipy import linalg
from scipy.linalg import LinAlgError
from scipy import weave
from scipy.optimize import fsolve
import copy
from operator import isSequenceType
from inspection import *
from units import second, mvolt
from clock import guess_clock
import magic
from equations import *
from itertools import count
from units import Quantity
import warnings
from log import *
from globalprefs import *
from experimental.codegen import *
CStateUpdater = PythonStateUpdater = None

def magic_state_updater(model, clock=None, order=1, implicit=False, compile=False, freeze=False, \
                        method=None, check_units=True):
    '''
    Examines the set of differential equations in 'model' (Equations object) and 
    returns a StateUpdater object and the list of dynamic variables.
    For example, the magic_state_updater function can determine if it
    is linear or nonlinear.
    
    Available methods:
    * None: the method is automatically selected
    * linear
    * Euler
    * RK (Runge-Kutta, second order)
    * exponential_Euler
    * nonlinear: automatic selection, but not linear
    '''
    global CStateUpdater, PythonStateUpdater
    if method == 'exponential_Euler':
        implicit = True
        order = 1
    elif method == 'Euler':
        implicit = False
        order = 1
    elif method == 'RK':
        implicit = False
        order = 2
    elif method is None:
        pass
    else:
        raise AttributeError, "Unknown integration method!"

    # All the first below should go in Equations
    if not(isinstance(model, Equations)): # a set of equations?
        raise TypeError, "An Equations object must be passed."

    model.prepare(check_units=check_units) # check units and other things
    dynamicvars = model._diffeq_names # Dynamic variables

    # Identify stochastic equations
    noiselist = []
    for statevar in model._diffeq_names:
        f = model._function[statevar]
        x0 = [model._units[var] for var in f.func_code.co_varnames] # init variables
        if depends_on(f, 'xi', x0):
            noiselist.append((statevar, get_global_term(f, 'xi', x0))) # s.d. of noise
            f.func_globals['xi'] = 0 * second ** -.5
        # better: remove in string

    use_codegen = get_global_preference('usecodegen') and get_global_preference('usecodegenstateupdate')
    use_weave = get_global_preference('useweave') and get_global_preference('usecodegenweave')
    if CStateUpdater is None:
        from experimental.codegen.stateupdaters import CStateUpdater, PythonStateUpdater

    # Linearity test
    # insert this in equations
    allow_linear = (method is None) or (method == 'linear')
    if allow_linear and model.is_linear():
        log_info('brian.stateupdater', "Linear model: using exact updates")
        stateupdaterobj = LinearStateUpdater(model, clock=clock)
    else:
        # Nonlinear model - check order of the method
        if implicit: # implicit integration schemes
            if model.is_conditionally_linear():
                log_info('brian.stateupdater', "Using exponential Euler")
                if not use_codegen:
                    stateupdaterobj = ExponentialEulerStateUpdater(model, clock=clock, compile=compile, freeze=freeze)
                elif use_weave:
                    stateupdaterobj = CStateUpdater(model, exp_euler_scheme, clock=clock, freeze=freeze)
                    log_warn('brian.stateupdater', 'Using codegen CStateUpdater')
                else:
                    stateupdaterobj = PythonStateUpdater(model, exp_euler_scheme, clock=clock, freeze=freeze)
                    log_warn('brian.stateupdater', 'Using codegen PythonStateUpdater')
            else:
                raise TypeError, "General implicit methods are not implemented yet."
        else: # explicit method
            if order == 1:
                if not use_codegen:
                    stateupdaterobj = NonlinearStateUpdater(model, clock=clock, compile=compile, freeze=freeze)
                elif use_weave:
                    stateupdaterobj = CStateUpdater(model, euler_scheme, clock=clock, freeze=freeze)
                    log_warn('brian.stateupdater', 'Using codegen CStateUpdater')
                else:
                    stateupdaterobj = PythonStateUpdater(model, euler_scheme, clock=clock, freeze=freeze)
                    log_warn('brian.stateupdater', 'Using codegen PythonStateUpdater')
            elif order == 2:
                if not use_codegen:
                    stateupdaterobj = RK2StateUpdater(model, clock=clock, compile=compile, freeze=freeze)
                elif use_weave:
                    stateupdaterobj = CStateUpdater(model, rk2_scheme, clock=clock, freeze=freeze)
                    log_warn('brian.stateupdater', 'Using codegen CStateUpdater')
                else:
                    stateupdaterobj = PythonStateUpdater(model, rk2_scheme, clock=clock, freeze=freeze)
                    log_warn('brian.stateupdater', 'Using codegen PythonStateUpdater')
            else:
                raise TypeError, "Methods with order greater than 2 are not implemented yet."

    # Insert noise
    for var, sigma in noiselist:
        # TODO: noise with mu = 0
        i = dynamicvars.index(var)
        stateupdaterobj = SynapticNoise(stateupdaterobj, i, 0 * model._units[var] / second, sigma, clock)

    return stateupdaterobj, dynamicvars

# TODO: StateUpdater should be lazy by default
class StateUpdater(object):
    '''
    A callable state update mechanism.
    By default, a leaky integrate-and-fire model with zero resting potential
    and unit time constant.
    Warning: to update the state matrix, use the slice operation, e.g.
    S[:]=0 (not S=0)
    otherwise operations are not done in place (a new object is created),
    so that all views are compromised (the reference to the data changes).
    '''
    def __init__(self, clock=None):
        '''
        Default model: dv/dt=-v
        '''
        if clock == None:
            self.update_factor = exp(-clock.dt) # The update matrix
        else:
            raise TypeError, "A time reference must be passed."

    def rest(self, P):
        '''
        Sets the variables at rest.
        P is the neuron group.
        '''
        warnings.warn('Rest is not implemented for this model')

    def __call__(self, P):
        '''
        Updates the state variables.
        Careful here: always use the slice operation for affectations.
        P is the neuron group.
        '''
        P._S[:] *= self.update_factor

    def __repr__(self):
        return 'Leaky integrate-and-fire StateUpdater'

    def __len__(self):
        '''
        Number of state variables
        '''
        return 1

#def get_linear_equations(eqs):
#    '''
#    Returns the matrices M and B for the linear model dX/dt = M(X-B),
#    where eqs is an Equations object. 
#    '''
#    # Otherwise assumes it is given in functional form
#    n=len(eqs._diffeq_names) # number of state variables
#    dynamicvars=eqs._diffeq_names
#    # Calculate B
#    AB=zeros((n,1))
#    d=dict.fromkeys(dynamicvars)
#    for j in range(n):
#        d[dynamicvars[j]]=0.*eqs._units[dynamicvars[j]]
#    for var,i in zip(dynamicvars,count()):
#        AB[i]=-eqs.apply(var,d)
#    # Calculate A
#    M=zeros((n,n))
#    for i in range(n):
#        for j in range(n):
#            d[dynamicvars[j]]=0.*eqs._units[dynamicvars[j]]
#        if isinstance(eqs._units[dynamicvars[i]],Quantity):
#            d[dynamicvars[i]]=Quantity.with_dimensions(1.,eqs._units[dynamicvars[i]].get_dimensions())
#        else:
#            d[dynamicvars[i]]=1.
#        for var,j in zip(dynamicvars,count()):
#            M[j,i]=eqs.apply(var,d)+AB[j]
#    M-=eye(n)*1e-10 # quick dirty fix for problem of constant derivatives; dimension = Hz
#    B=linalg.lstsq(M,AB)[0] # We use this instead of solve in case M is degenerate
#    return M,B

def get_linear_equations(eqs):
    '''
    Returns the matrices M and B for the linear model dX/dt = M(X-B),
    where eqs is an Equations object. 
    '''
    # Otherwise assumes it is given in functional form
    n = len(eqs._diffeq_names) # number of state variables
    dynamicvars = eqs._diffeq_names
    # Calculate B
    AB = zeros((n, 1))
    d = dict.fromkeys(dynamicvars)
    for j in range(n):
        d[dynamicvars[j]] = 0. * eqs._units[dynamicvars[j]]
    for var, i in zip(dynamicvars, count()):
        AB[i] = -eqs.apply(var, d)
    # Calculate A
    M = zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[dynamicvars[j]] = 0. * eqs._units[dynamicvars[j]]
        if isinstance(eqs._units[dynamicvars[i]], Quantity):
            d[dynamicvars[i]] = Quantity.with_dimensions(1., eqs._units[dynamicvars[i]].get_dimensions())
        else:
            d[dynamicvars[i]] = 1.
        for var, j in zip(dynamicvars, count()):
            M[j, i] = eqs.apply(var, d) + AB[j]
    #M-=eye(n)*1e-10 # quick dirty fix for problem of constant derivatives; dimension = Hz
    #B=linalg.lstsq(M,AB)[0] # We use this instead of solve in case M is degenerate
    B = linalg.solve(M, AB) # We use this instead of solve in case M is degenerate
    return M, B

def get_linear_equations_solution_numerically(eqs, dt):
    # Otherwise assumes it is given in functional form
    n = len(eqs._diffeq_names) # number of state variables
    dynamicvars = eqs._diffeq_names
    # Calculate B
    AB = zeros((n, 1))
    d = dict.fromkeys(dynamicvars)
    for j in range(n):
        d[dynamicvars[j]] = 0. * eqs._units[dynamicvars[j]]
    for var, i in zip(dynamicvars, count()):
        AB[i] = -eqs.apply(var, d)
    # Calculate A
    M = zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[dynamicvars[j]] = 0. * eqs._units[dynamicvars[j]]
        if isinstance(eqs._units[dynamicvars[i]], Quantity):
            d[dynamicvars[i]] = Quantity.with_dimensions(1., eqs._units[dynamicvars[i]].get_dimensions())
        else:
            d[dynamicvars[i]] = 1.
        for var, j in zip(dynamicvars, count()):
            M[j, i] = eqs.apply(var, d) + AB[j]
    #B=linalg.solve(M,AB)
    numeulersteps = 100
    deltat = dt / numeulersteps
    E = eye(n) + deltat * M
    C = eye(n)
    D = zeros((n, 1))
    for step in xrange(numeulersteps):
        C, D = dot(E, C), dot(E, D) - AB * deltat
    return C, D
    #return M,B

set_global_preferences(useweave_linear_diffeq=False)
define_global_preference('useweave_linear_diffeq', 'False',
                           desc="""
                                  Whether to use weave C++ acceleration for the solution
                                  of linear differential equations. Note that on some
                                  platforms, typically older ones, this is faster and on
                                  some platforms, typically new ones, this is actually
                                  slower.
                                  """)


class LinearStateUpdater(StateUpdater):
    '''
    A linear model with dynamics dX/dt = M(X-B) or dX/dt = MX.
    
    **Initialised as:** ::
    
        LinearStateUpdater(M[,B[,clock]])
    
    with arguments:
    
    ``M``
        Matrix defining the differential equation.
    ``B``
        Optional linear term in the differential equation.
    ``clock``
        Optional clock.
    
    Computes an update matrix A=exp(M dt) for the linear system,
    and performs the update step.
    
    TODO: more mathematical details? 
    '''
    #TODO: sparse linear models (e.g. cable equations)
    def __init__(self, M, B=None, clock=None):
        '''
        Initialize a linear model with dynamics dX/dt = M(X-B) or dX/dt = MX,
        where B is a column vector.
        TODO: more checks
        TODO: rest
        '''
        self._useaccel = get_global_preference('useweave_linear_diffeq')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._cpp_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
        self._useB = False
        if clock == None:
            clock = guess_clock()
        if isinstance(M, ndarray):
            self.A = linalg.expm(M * clock.dt)
            self.B = B
        elif isinstance(M, Equations):
            try:
                M, self.B = get_linear_equations(M)
                self.A = linalg.expm(M * clock.dt)
                #self.A=array(self.A,single)
                if self.B is not None:
                    self._C = -dot(self.A, self.B) + self.B
                    #self._C=array(self._C,single)
                    self._useB = True
                else:
                    self._useB = False
            except LinAlgError:
                log_info('brian.stateupdater', 'Solving linear equations numerically')
                self.A, self._C = get_linear_equations_solution_numerically(M, clock.dt)
                self.B = NotImplemented # raises error on trying to use this
                self._useB = True
        # note the numpy dot command works faster if self.A has C ordering compared
        # to fortran ordering (although maybe this depends on which implementation
        # of BLAS you're using). The difference is only significant in small
        # calculations because making a copy of self.A is usually not serious, its
        # size is only the number of variables, not the number of neurons.
        self.A = array(self.A, order='C')
        if self._useB:
            self._C = array(self._C, order='C')

    def rest(self, P):
        if self._useB:
            if self.B is NotImplemented:
                raise NotImplementedError, \
                    "The resting potential cannot be found because the equations are degenerate " + \
                    "(most likely because they include a parameter)"
            P._S[:] = self.B
        else:
            P._S[:] = 0

    def __call__(self, P):
        '''
        Updates the state variables.
        Careful here: always use the slice operation for affectations.
        P is the neuron group.
        '''
        if self._useB: # This could be removed
            if not self._useaccel:
                #P._S[:]=dot(self.A,P._S)+self._C
                P._S[:] = dot(self.A, P._S)
                #P._S = dot(self.A,P._S)
                #P._S += self._C
                add(P._S, self._C, P._S)
                #P._S[:]=dot(self.A,P._S-self.B)+self.B
            else:
                m = len(self)
                S = P._S
                n = S.shape[1] #n = len(P)
                A = self.A
                c = self._C
                code = '''
                double x[m]; 
                for(int i=0;i<n;i++)  
                {
                    for(int j=0;j<m;j++)
                    {
                        x[j] = c(j);
                        for(int k=0;k<m;k++)
                           x[j] += A(j,k) * S(k,i);
                    }
                    for(int j=0;j<m;j++)
                        S(j,i) = x[j];
                } 
                '''
                weave.inline(code, ['n', 'm', 'S', 'A', 'c'],
                             compiler=self._cpp_compiler,
                             type_converters=weave.converters.blitz,
                             extra_compile_args=self._extra_compile_args)
        else:
            if not self._useaccel:
                P._S[:] = dot(self.A, P._S)
            else:
                n = len(P)
                m = len(self)
                S = P._S
                A = self.A
                code = '''
                double x[m]; 
                for(int i=0;i<n;i++)  
                {
                    for(int j=0;j<m;j++)
                    {
                        x[j] = 0.0;
                        for(int k=0;k<m;k++)
                           x[j] += A(j,k) * S(k,i);
                    }
                    for(int j=0;j<m;j++)
                        S(j,i) = x[j];
                } 
                '''
                weave.inline(code, ['n', 'm', 'S', 'A'],
                             compiler=self._cpp_compiler,
                             type_converters=weave.converters.blitz,
                             extra_compile_args=self._extra_compile_args)

    def __repr__(self):
        return 'Linear StateUpdater with ' + str(len(self)) + ' state variables'

    def __len__(self):
        '''
        Number of state variables
        '''
        return self.A.shape[0]


class NonlinearStateUpdater(StateUpdater):
    '''
    A nonlinear model with dynamics dX/dt = f(X).
    Uses an Equations object.
    By default, uses Euler integration.
    '''
    def __init__(self, eqs, clock=None, compile=False, freeze=False):
        '''
        Initialize a nonlinear model with dynamics dX/dt = f(X).
        f is given as an Equations object (see examples).
        If compile is True, a Python code object is compiled.
        '''
        # TODO: global pref?
        self.eqs = eqs
        self.optimized = compile
        self._first_time = True
        if freeze:
            self.eqs.compile_functions(freeze=freeze)
        if compile:
            self._code = self.eqs.forward_euler_code()

    def rest(self, P):
        '''
        Sets the variables at rest.
        '''
        for name, value in self.eqs.fixed_point().iteritems():
            P.state(name)[:] = value

    def __call__(self, P):
        '''
        Updates the state variables.
        Careful here: always use the slice operation for affectations.
        P is the neuron group.
        Euler integration.
        '''
        #if self.optimized==False:
        #    self.eqs.optimize(len(P))
        #    self.optimized=True
        # TODO: do these operations once
        #states=dict.fromkeys(self.eqs.dynamicvars)
        # store that in the neurongroup?
        if self.optimized:
            if self._first_time:
                self._first_time = False
                P._dS = 0 * P._S
            dt = P.clock._dt
            t = P.clock.t
            exec(self._code)
        else:
            states = dict.fromkeys(self.eqs._diffeq_names) # ={}?
            #for var in self.eqs.dynamicvars:
            for var in self.eqs._diffeq_names:
                states[var] = P.state_(var)
            states['t'] = P.clock.t #time
            self.eqs.forward_euler(states, P.clock._dt)

    def __repr__(self):
        return 'Nonlinear StateUpdater with ' + str(len(self)) + ' state variables'

    def __len__(self):
        '''
        Number of state variables
        '''
        return len(self.eqs)


class RK2StateUpdater(NonlinearStateUpdater):
    '''
    A nonlinear model with dynamics dX/dt = f(X).
    Uses an Equations object.
    Uses Runge-Kutta midpoint integration (second order).
    '''
    def __init__(self, eqs, clock=None, compile=False, freeze=False):
        '''
        Initialize a nonlinear model with dynamics dX/dt = f(X).
        f is given as an Equations object (see examples).
        If compile is True, a Python code object is compiled.
        '''
        # TODO: global pref?
        self.eqs = eqs
        self._first_time = True
        if freeze:
            self.eqs.compile_functions(freeze=freeze)
        if compile:
            warnings.warn('Compilation is not implemented yet for RK2 integration.')

    def __call__(self, P):
        '''
        Updates the state variables.
        Careful here: always use the slice operation for affectations.
        P is the neuron group.
        Euler integration.
        '''
        states = dict.fromkeys(self.eqs._diffeq_names) # ={}?
        #for var in self.eqs.dynamicvars:
        for var in self.eqs._diffeq_names:
            states[var] = P.state_(var)
        states['t'] = P.clock.t #time
        self.eqs.Runge_Kutta2(states, P.clock._dt)


class ExponentialEulerStateUpdater(NonlinearStateUpdater):
    def __init__(self, eqs, clock=None, compile=False, freeze=False):
        '''
        Initialize a nonlinear model with dynamics dX/dt = f(X).
        '''
        # TODO: global pref?
        self.eqs = eqs
        self.optimized = compile
        self._first_time = True
        if freeze:
            self.eqs.compile_functions(freeze=freeze)
        if compile:
            self._code = self.eqs.exponential_euler_code()

    def __call__(self, P):
        '''
        Updates the state variables.
        Careful here: always use the slice operation for affectations.
        P is the neuron group.
        '''
        if self.optimized:
            if self._first_time:
                self._first_time = False
                P._dS = P._S.copy()
            dt = P.clock._dt
            t = P.clock.t
            exec(self._code)
        else:
            # TODO: do these operations once
            states = dict.fromkeys(self.eqs._diffeq_names) #={}?
            for var in self.eqs._diffeq_names:
                states[var] = P.state_(var)
            states['t'] = P.clock.t #time
            self.eqs.exponential_euler(states, P.clock._dt)


class SynapticNoise(StateUpdater):
    '''
    Synaptic noise mechanism, plugged into another StateUpdater.
    '''
    def __init__(self, baseupdater, nstate, mu, sigma, clock=None):
        '''
        baseupdater = source neuron StateUpdater
        nstate = index of synaptic state variable
        mu = mean synaptic input rate (per ms)
        sigma = s.d. of synaptic input per ms^{1/2}
        '''
        self.baseupdater = baseupdater
        self.nstate = nstate
        if clock == None:
            clock = guess_clock()
        if clock:
            # TODO: check units
            self.mu = mu * clock.dt
            self.sigma = sigma * clock.dt ** .5
        else:
            raise TypeError, "A time reference must be passed."

    def rest(self, P):
        self.baseupdater.rest(P)

    def __call__(self, P):
        '''
        Updates the state variables.
        Careful here: always use the slice operation for affectations.
        P is the neuron group.
        '''
        self.baseupdater(P) # update the underlying model
        P._S[self.nstate, :] += self.mu + random.randn(P._S.shape[1]) * self.sigma

    def __repr__(self):
        return self.baseupdater.__repr__() + ' with synaptic noise on variable ' + str(self.nstate)

    def __len__(self):
        '''
        Number of state variables
        '''
        return len(self.baseupdater)


class LazyStateUpdater(StateUpdater):
    '''
    A StateUpdater that does nothing.
    
    **Initialised as:** ::
    
        LazyStateUpdater([numstatevariables=1[,clock]])
        
    with arguments:
    
    ``numstatevariables``
        The number of state variables to create.
    ``clock``
        An optional clock to determine when it updates,
        although the update function does nothing so...
    '''
#    Alternatively, we might replace the parent StateUpdater class by this and
#    write a basic leaky integrator class.
    def __init__(self, numstatevariables=1, clock=None):
        self._N = numstatevariables
        pass

    def __call__(self, P):
        '''
        Updates the state variables.
        '''
        pass

    def __repr__(self):
        return 'Lazy StateUpdater (does nothing)'

    def __len__(self):
        '''
        Number of state variables
        '''
        return self._N

# UNTESTED
class FunStateUpdater(StateUpdater):
    """
    A StateUpdater that calls a function at each update step
    
    A StateUpdater function takes one argument, the neuron group
    that is being updated.
    """
    def __init__(self, func, numstates, clock=None):
        self.clock = guess_clock(clock)
        self.func = func
        self.numstates = numstates

    def __call__(self, P):
        self.func(P)

    def __repr__(self):
        return "Function StateUpdater, function " + self.func.__name__

    def __len__(self):
        return self.numstates
