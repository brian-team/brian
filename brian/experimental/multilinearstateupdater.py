from brian import *
from brian.stateupdater import get_linear_equations_solution_numerically, get_linear_equations
from scipy import linalg, weave
import numpy

__all__=['MultiLinearStateUpdater','get_multilinear_state_updater','MultiLinearNeuronGroup']

def get_multilinear_state_updater(eqs, subs, level=0, clock=None):
    '''
    Make a multilinear state updater
    
    Arguments:
    
    ``eqs``
        should be the equations, and must be a string not an :class:`Equations` object.
    ``subs``
        A list of dictionaries, each dictionary gives the variable substitutions
        to make. There should be one dictionary for each neuron in the final
        group.
    ``level``
        How many levels up to look for the equations' namespace.
    ``clock``
        If you want.
    '''
    AiBi = []
    useB = False
    for s in subs:
        neweqs = eqs
        for k, v in s.items():
            neweqs = neweqs.replace(k,'('+str(v)+')')
        neweqs = Equations(neweqs, level=level+1)
        Ai, Bi = get_linear_matrices(neweqs, clock=clock)
        AiBi.append((Ai, Bi))
        useB = useB or Bi is not None
        n = Ai.shape[0]
    A = numpy.zeros((n, n, len(subs)))
    if useB:
        B = numpy.zeros((n, len(subs)))
    else:
        B = None
    for i, (Ai, Bi) in enumerate(AiBi):
        A[:,:,i] = Ai
        if useB and Bi is not None:
            B[:,i] = Bi.squeeze()
    return MultiLinearStateUpdater(A, B)

def get_linear_matrices(eqs, clock=None):
    '''
    This is just a copy of what the main Brian linear equations code does, but self-contained
    '''
    eqs.prepare()
    if clock==None:
        clock = guess_clock()
    try:
        M, C = get_linear_equations(eqs)
        A = linalg.expm(M*clock.dt)
        if C is not None:
            B = -dot(A,C)+C
        else:
            B = None
        return A, B
    except LinAlgError:
        A, B = get_linear_equations_solution_numerically(eqs, clock.dt)
        return A, B

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
            weave.inline(code,['n','m','S','A','B'],
                         compiler=self._cpp_compiler,
                         type_converters=weave.converters.blitz,
                         extra_compile_args=['-O3'])
        else:
            if self.A.shape[2]<self.A.shape[1]:
                for i in xrange(self.A.shape[2]):
                    P._S[:,i] = dot(self.A[:,:,i], P._S[:,i])
            else:
                # this is equivalent to the above but the loop is smaller if the
                # number of neurons is large
                AS = self.A[:,0,:]*P._S[0,:]
                for i in xrange(1, self.A.shape[1]):
                    AS += self.A[:,i,:]*P._S[i,:]
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
    
    You give a single set of differential equations with parameters, and you
    also give a list of substitutions for those parameters, one set of
    substitutions for each neuron in the group.
    
    Arguments:
    
    ``eqs``
        should be the equations, and must be a string not an :class:`Equations` object.
    ``subs``
        A list of dictionaries, each dictionary gives the variable substitutions
        to make. There should be one dictionary for each neuron in the final
        group.
    ``level``
        How many levels up to look for the equations' namespace.
    ``clock``
        If you want.
    '''
    def __init__(self, eqs, subs, clock=None, level=0, **kwds):
        neweqs = eqs
        for k, v in subs[0].items():
            neweqs = neweqs.replace(k,'('+str(v)+')')
        neweqs = Equations(neweqs, level=level+1)
        NeuronGroup.__init__(self, len(subs), neweqs, clock=clock, **kwds)
        self._state_updater = get_multilinear_state_updater(eqs, subs, level=level+1)     
    
if __name__=='__main__':
    eqs = '''
    dv/dt = k*v/(1*second) : 1
    dw/dt = k*w/(1*second) : 1
    '''
    subs = [{'k':-1},
            {'k':-2},
            {'k':-3}]
    G = MultiLinearNeuronGroup(eqs, subs)
    G.v = 1
    G.w = 0
    M = StateMonitor(G, 'v', record=True)
    run(1*second)
    for i in range(len(G)):
        plot(M.times, M[i])
    show()