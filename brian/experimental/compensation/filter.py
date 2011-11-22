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

def compute_filter(A, row=0):
    d = len(A)

    # compute a
    a = poly(A)  # directly vector a of the filter, a[0]=1

    # compute b recursively
    b = zeros(d+1)
    T = eye(d)
    b[0] = T[row, 0]
    for i in range(1, d+1):
        T = a[i]*eye(d) + dot(A, T)
        b[i] = T[row, 0]

    return b, a

def simulate(eqs, I, dt, row=0):
    """
    I must be normalized (I*Re/taue for example)
    """
    M, B = get_linear_equations(eqs)
    A = linalg.expm(M * dt)
    b, a = compute_filter(A, row=row)
    y = lfilter(b, a, I*dt) + B[row]
    return y

if __name__ == '__main__':
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


    

    Is = numpy.load("current1.npy")[:50000]
    eqs2 = Equations("""
        dV/dt=Re*(I-Iinj)/taue : volt
        dV0/dt=(R*Iinj-V0+Vr)/tau : volt
        Iinj=(V-V0)/R : amp
        I : amp
    """)

    G = NeuronGroup(1, eqs2)
    G.V = Vr
    G.I = TimedArray(Is)
    stm = StateMonitor(G, 'V', record=True)
    t0 = time.clock()
    run(len(Is)*dt)
    t1 = time.clock()-t0
    y0 = stm.values[0]

    t0 = time.clock()
    y = simulate(eqs, Is * Re/taue, dt, row=0)
    t2 = time.clock()-t0

    print t1, t2, t1/t2
    print max(abs(y-y0))

    subplot(211)
    plot(Is)
    subplot(212)
    plot(y0)
    plot(y)
    show()
