import brian_no_units
from brian import *
from itertools import count
from scipy.signal import lfilter
from scipy import linalg
from numpy.linalg import inv, matrix_power
import numpy
import time

def get_linear_equations(eqs):
    '''
    Returns the matrices A and C for the linear model dX/dt = M(X-B),
    where eqs is an Equations object. 
    '''
    # Otherwise assumes it is given in functional form
    n = len(eqs._diffeq_names) # number of state variables
    dynamicvars = eqs._diffeq_names
    
    #print eqs._diffeq_names

    # Calculate B
    AB = zeros((n, 1))
    d = dict.fromkeys(dynamicvars)

    #print d

    for j in range(n):
        d[dynamicvars[j]] = 0. * eqs._units[dynamicvars[j]]
    for var, i in zip(dynamicvars, count()):
        #print i, var
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
    
    try:
        B = linalg.solve(M, AB) # We use this instead of solve in case M is degenerate
    except:
        B = None

    return M, B

def compute_filter(A, U=None):
    """
    Returns the lfilter (b,a) from the system X(n+1)=A X(n) + B(n)
    To simulate the system : y = lfilter(b, a, x) where B(n)=x[n]*U.
    u is a vector, by default it is [1, 0,...,0], but it can be any vector
    so that the input x is injected with different scales in the different 
    variables.
    All the initial conditions must be 0.
    """
    d = len(A)

    #if rows == 'all':
    #    rows = range(d)
        
    if U is None:
        U = zeros(d)
        U[0] = 1.
    U = array(U).flatten()

    # compute a
    a = poly(A)  # directly vector a of the filter, a[0]=1

    # compute b recursively
    b = zeros(d+1)

    T = eye(d)
    for k in range(1, d+1):
        T = a[k]*eye(d) + dot(A, T)
        b[k] = sum(T[0, :] * U)

    return b, a

def simulate(eqs, I, dt, U=None):
    """
    I must be normalized (I*Re/taue for example)
    """
    M, B = get_linear_equations(eqs)
    A = linalg.expm(M * dt)
    b, a = compute_filter(A, U=U)
    y = lfilter(b, a, I) + B[0]
    return y

def test_simulate():
    R = 500*Mohm
    Re = 400*Mohm
    tau = 10*ms
    taue = 1.0*ms
    Vr = -70*mV
    dt = .1*ms

    # +Re*I/taue
    eqs = Equations("""
        dV/dt=Re*(-Iinj)/taue : volt
        dV0/dt=(R*Iinj-V0+Vr)/tau : volt
        Iinj=(V-V0)/R : amp
    """)
    eqs.prepare()


    

    Is = numpy.load("current1.npy")[:10000]
    eqs2 = Equations("""
        dV/dt=Re*(I-Iinj)/taue : volt
        dV0/dt=(R*Iinj-V0+Vr)/tau : volt
        Iinj=(V-V0)/R : amp
        I : amp
    """)

    G = NeuronGroup(1, eqs2)
    G.V = G.V0 = Vr
    G.I = TimedArray(Is)
    stm = StateMonitor(G, 'V0', record=True)
    t0 = time.clock()
    run(len(Is)*defaultclock.dt)
    t1 = time.clock()-t0
    y0 = stm.values[0]

    t0 = time.clock()
    U = array([Re/taue*dt, 0]) 
    y = simulate(eqs, Is, defaultclock.dt, U=U)
    t2 = time.clock()-t0

    print t1, t2, t1/t2
    print max(abs(y-y0))

    subplot(211)
    plot(Is)
    subplot(212)
    plot(y0)
    plot(y)
    show()
    
if __name__ == '__main__':
    test_simulate()
