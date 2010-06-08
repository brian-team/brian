from brian import *
from brian.stateupdater import get_linear_equations_solution_numerically, get_linear_equations
from scipy import linalg, weave
import numpy
import inspect
from itertools import count
import re

__all__ = ['MultiLinearStateUpdater', 'get_multilinear_state_updater', 'MultiLinearNeuronGroup']

__all__ = ['MultiLinearStateUpdater', 'get_multilinear_state_updater', 'MultiLinearNeuronGroup']

def get_multilinear_matrices(eqs, subs, clock=None):
    '''
    Returns the matrices M and B for the linear model dX/dt = M(X-B),
    where eqs is an Equations object. 
    '''
    if clock is None: clock = guess_clock()
    nsubs = len(subs[subs.keys()[0]])
    # Otherwise assumes it is given in functional form
    n = len(eqs._diffeq_names) # number of state variables
    dynamicvars = eqs._diffeq_names
    # Calculate B
    AB = zeros((n, 1, nsubs))
    d = dict.fromkeys(dynamicvars)
    for j in range(n):
        if dynamicvars[j] in subs:
            d[dynamicvars[j]] = subs[j]
        else:
            d[dynamicvars[j]] = 0. * eqs._units[dynamicvars[j]]
    for var, i in zip(dynamicvars, count()):
        AB[i, 0, :] = -eqs.apply(var, d)
    # Calculate A
    M = zeros((n, n, nsubs))
    for i in range(n):
        for j in range(n):
            if dynamicvars[j] in subs:
                d[dynamicvars[j]] = subs[j]
            else:
                d[dynamicvars[j]] = 0. * eqs._units[dynamicvars[j]]
        if dynamicvars[i] in subs:
            d[dynamicvars[i]] = subs[i]
        else:
            if isinstance(eqs._units[dynamicvars[i]], Quantity):
                d[dynamicvars[i]] = Quantity.with_dimensions(1., eqs._units[dynamicvars[i]].get_dimensions())
            else:
                d[dynamicvars[i]] = 1.
        for var, j in zip(dynamicvars, count()):
            M[j, i, :] = eqs.apply(var, d) + AB[j]

    allM = M
    allAB = AB

    AiBi = []
    for i in xrange(nsubs):
        try:
            M = reshape(allM[:, :, i], allM.shape[:2])
            AB = reshape(allAB[:, :, i], allAB.shape[:2])
            C = linalg.solve(M, AB)
            A = linalg.expm(M * clock.dt)
            B = -dot(A, C) + C
        except LinAlgError:
            numeulersteps = 100
            deltat = clock.dt / numeulersteps
            E = eye(n) + deltat * M
            C = eye(n)
            D = zeros((n, 1))
            for step in xrange(numeulersteps):
                C, D = dot(E, C), dot(E, D) - AB * deltat
            A, B = C, D
        AiBi.append((A, B))

    return AiBi

def get_multilinear_state_updater(eqs, subs, level=0, clock=None):
    '''
    Make a multilinear state updater
    
    Arguments:
    
    ``eqs``
        should be the equations, and must be a string not an :class:`Equations` object.
    ``subs``
        A dictionary of arrays, each key k is a variable name, each value is an equally
        sized array of values it takes, this array should have the size of the number
        of neurons in the group.
    ``level``
        How many levels up to look for the equations' namespace.
    ``clock``
        If you want.
    '''
    neweqs = Equations(eqs, level=level + 1)
    neweqs.prepare()
    AiBi = get_multilinear_matrices(neweqs, subs, clock=clock)
    n = AiBi[0][0].shape[0]
    nsubs = len(subs[subs.keys()[0]])
    A = numpy.zeros((n, n, nsubs))
    B = numpy.zeros((n, nsubs))
    for i, (Ai, Bi) in enumerate(AiBi):
        A[:, :, i] = Ai
        B[:, i] = Bi.squeeze()
    return MultiLinearStateUpdater(A, B)


class MultiLinearStateUpdater(StateUpdater):
    '''
    A StateUpdater with one differential equation for each neuron
    
    Initialise with:
    
    ``A``, ``B``
        Arrays, see below.
        
    The update step for neuron i is S[:,i] = dot(A[:,:,i], S[:,i]) + B[:, i]. 
    '''
    def __init__(self, A, B=None):
        self.A = A
        self.B = B
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._cpp_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']

    def __call__(self, P):
        if self._useaccel:
            n = len(P)
            m = len(self)
            S = P._S
            A = self.A
            B = self.B
            code = '''
            double x[m]; 
            for(int i=0;i<n;i++)  
            {
                for(int j=0;j<m;j++)
                {
                    x[j] = B(j,i);
                    for(int k=0;k<m;k++)
                       x[j] += A(j,k,i) * S(k,i);
                }
                for(int j=0;j<m;j++) 
                    S(j,i) = x[j];
            } 
            '''
            weave.inline(code, ['n', 'm', 'S', 'A', 'B'],
                         compiler=self._cpp_compiler,
                         type_converters=weave.converters.blitz,
                         extra_compile_args=self._extra_compile_args)
        else:
            if self.A.shape[2] < self.A.shape[1]:
                for i in xrange(self.A.shape[2]):
                    P._S[:, i] = dot(self.A[:, :, i], P._S[:, i])
            else:
                # this is equivalent to the above but the loop is smaller if the
                # number of neurons is large
                AS = self.A[:, 0, :]*P._S[0, :]
                for i in xrange(1, self.A.shape[1]):
                    AS += self.A[:, i, :]*P._S[i, :]
                P._S[:] = AS
            if self.B is not None:
                add(P._S, self.B, P._S)

    def __len__(self):
        return self.A.shape[0]

    def __repr__(self):
        return 'MultiLinearStateUpdater'
    __str__ = __repr__


class MultiLinearNeuronGroup(NeuronGroup):
    '''
    Make a NeuronGroup with a linear differential equation for each neuron
    
    You give a single set of differential equations with parameters, the
    variables you want substituted should be defined as parameters in the equations,
    but they will not be treated as parameters, instead they will be substituted.
    You also pass a list of variables to have their values substituted, and these
    names should exist in the namespace initialising the MultiLinearNeuronGroup. 
    
    Arguments:
    
    ``eqs``
        should be the equations, and must be a string not an :class:`Equations` object.
    ``subs``
        A list of variables to be substituted with values.
    ``level``
        How many levels up to look for the equations' namespace.
    ``clock``
        If you want.
    ``kwds``
        Any additonal arguments to pass to :class:`NeuronGroup` init.
    '''
    #TODO: for consistency put the units in the equations, and multilinearneurongroup can extract them.
    def __init__(self, eqs, subs, clock=None, level=0, **kwds):
        subs_set = set(subs)
        param_pattern = re.compile('\s*(\w+)\s*:\s*(.*)')
        eqs = re.sub('\\\s*?\n', ' ', eqs) # compact multiline equations
        neweqs = ''
        subs = {}
        for line in eqs.splitlines():
            line = re.sub('#.*', '', line) # remove comments
            result = param_pattern.match(line)
            added = False
            if result is not None:
                name, unit = result.groups()
                if name in subs_set:
                    subs[name] = unit
                    added = True
            if not added:
                neweqs += line + '\n'
        eqs = neweqs
        frame = inspect.stack()[level + 1][0]
        ns_global, ns_local = frame.f_globals, frame.f_locals
        k0 = subs.keys()[0]
        if k0 in ns_local:
            nsubs = len(ns_local[k0])
        else:
            nsubs = len(ns_global[k0])
        neweqs = eqs
        for k, u in subs.iteritems():
            neweqs = neweqs.replace(k, '(0.*(' + u + '))')
        neweqs = Equations(neweqs, level=level + 1)
        NeuronGroup.__init__(self, nsubs, neweqs, clock=clock, **kwds)
        subs2 = {}
        for k, u in subs.iteritems():
            if k in ns_local:
                subs2[k] = ns_local[k]
            else:
                subs2[k] = ns_global[k]
        self._state_updater = get_multilinear_state_updater(eqs, subs2, level=level + 1, clock=clock)

if __name__ == '__main__':
    eqs = '''
    dv/dt = k*v/(1*second) : 1
    dw/dt = k*w/(1*second) : 1
    k : 1
    '''
    k = array([-1, -2, -3])
    subs = ['k']
    G = MultiLinearNeuronGroup(eqs, subs)
    G.v = 1
    G.w = 0
    M = StateMonitor(G, 'v', record=True)
    run(1 * second)
    for i in range(len(G)):
        plot(M.times, M[i])
    show()
