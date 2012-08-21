"""
Electrode compensation
----------------------
Active Electrode Compensation (AEC)
"""
from brian.stateupdater import get_linear_equations
from brian.log import log_info
from brian import second, Mohm, mV, ms, Equations, ohm, volt, second
from scipy.optimize import fmin
from scipy.signal import lfilter
from scipy import linalg
from numpy import sqrt, ceil, zeros, eye, poly, dot, hstack, array
from scipy import zeros, array, optimize, mean, arange, diff, rand, exp, sum, convolve, eye, linalg, sqrt
import time

__all__=['full_kernel', 'full_kernel_from_step',
         'electrode_kernel_soma', 'electrode_kernel_dendrite', 'solve_convolution',
         'electrode_kernel', 'AEC_compensate']

'''
Active Electrode Compensation
-----------------------------
From:
Brette et al (2008). High-resolution intracellular recordings using a real-time
computational model of the electrode. Neuron 59(3):379-91.
'''
def full_kernel(v, i, ksize, full_output=False):
    '''
    Calculates the full kernel from the recording v and the input
    current i. The last ksize steps of v should be null.
    ksize = size of the resulting kernel
    full_output = returns K,v0 if True (v0 is the resting potential)
    '''
    # Calculate the correlation vector <v(n)i(n-k)>
    # and the autocorrelation vector <i(n)i(n-k)>
    vi = zeros(ksize)
    ii = zeros(ksize)
    vref = mean(v) # taking <v> as the reference potential simplifies the formulas
    for k in range(ksize):
        vi[k] = mean((v[k:] - vref) * i[:len(i) - k])
        ii[k] = mean(i[k:] * i[:len(i) - k])
    vi -= mean(i) ** 2
    K = levinson_durbin(ii, vi)
    if full_output:
        v0 = vref - mean(i) * sum(K)
        return K, v0
    else:
        return K

def full_kernel_from_step(V, I):
    '''
    Calculates the full kernel from the response (V) to a step input
    (I, constant).
    '''
    return diff(V) / I

def solve_convolution(K, Km):
    '''
    Solves Ke = K - Km * Ke/Re
    Linear problem
    '''
    Re = sum(K) - sum(Km)
    n = len(Km)
    A = eye(n) * (1 + Km[0] / Re)
    for k in range(n):
        for m in range(k):
            A[k, m] = Km[k - m] / Re
    return linalg.lstsq(A, K)[0]

def electrode_kernel_dendrite(Karg, start_tail, full_output=False):
    '''
    (For dendritic recordings)
    Extracts the electrode kernel Ke from the raw kernel K
    by removing the membrane kernel, estimated from the
    indexes >= start_tail of the raw kernel.
    full_output = returns Ke,Km if True (otherwise Ke)
    (Ke=electrode filter, Km=membrane filter)
    '''

    K = Karg.copy()

    def remove_km(RawK, Km):
        '''
        Solves Ke = RawK - Km * Ke/Re for a dendritic Km.
        '''
        Kel = RawK - Km
        # DOES NOT CONVERGE!!
        for _ in range(5): # Iterative solution
            Kel = RawK - convolve(Km, Kel)[:len(Km)] / sum(Kel)
            # NB: Re=sum(Kel) increases after every iteration
        return Kel

    # Fit of the tail to a dendritic kernel to find the membrane time constant
    t = arange(len(K))
    tail = arange(start_tail, len(K))
    Ktail = K[tail]
    f = lambda params:params[0] * ((tail + 1) ** -.5) * exp(-params[1] ** 2 * (tail + 1)) - Ktail
    #Rtail=sum(Ktail)
    #g=lambda tau:sum((tail+1)**(-.5)*exp(-(tail+1)/tau))
    #J=lambda tau:sum(((tail+1)**(-.5)*exp(-(tail+1)/tau)/g(tau)-Ktail/Rtail)**2)
    p, _ = optimize.leastsq(f, array([1., .03]))
    #p=optimize.fminbound(J,.1,10000.)
    #p=optimize.golden(J)

    #print "tau_dend=",p*.1
    #Km=(t+1)**(-.5)*exp(-(t+1)/p)*Rtail/g(p)

    print "tau_dend=", .1 / (p[1] ** 2)
    Km = p[0] * ((t + 1) ** -.5) * exp(-p[1] ** 2 * (t + 1))
    K[tail] = Km[tail]

    # Find the minimum
    z = optimize.fminbound(lambda x:sum(solve_convolution(K, x * Km)[tail] ** 2), .5, 1.)
    Ke = solve_convolution(K, z * Km)

    if full_output:
        return Ke[:start_tail], z * Km
    else:
        return Ke[:start_tail]

def electrode_kernel_soma(Karg, start_tail, full_output=False):
    '''
    (For somatic recordings - alternative method)
    Extracts the electrode kernel Ke from the raw kernel K
    by removing the membrane kernel, estimated from the
    indexes >= start_tail of the raw kernel.
    full_output = returns Ke,Km if True (otherwise Ke)
    (Ke=electrode filter, Km=membrane filter)
    '''

    K = Karg.copy()

    def remove_km(RawK, Km):
        '''
        Solves Ke = RawK - Km * Ke/Re for a dendritic Km.
        '''
        Kel = RawK - Km
        for _ in range(5): # Iterative solution
            Kel = RawK - convolve(Km, Kel)[:len(Km)] / sum(Kel)
            # NB: Re=sum(Kel) increases after every iteration
        return Kel

    # Fit of the tail to a somatic kernel to find the membrane time constant
    t = arange(len(K))
    tail = arange(start_tail, len(K))
    Ktail = K[tail]
    f = lambda params:params[0] * exp(-params[1] ** 2 * (tail + 1)) - Ktail
    p, _ = optimize.leastsq(f, array([1., .3]))
    Km = p[0] * exp(-p[1] ** 2 * (t + 1))
    print "tau_soma=", .1 / (p[1] ** 2)

    K[tail] = Km[tail]

    # Find the minimum
    z = optimize.fminbound(lambda x:sum(solve_convolution(K, x * Km)[tail] ** 2), .5, 1.)
    Ke = solve_convolution(K, z * Km)
    print "R=", sum(z * p[0] * exp(-p[1] ** 2 * (arange(1000) + 1)))

    if full_output:
        return Ke[:start_tail], z * Km
    else:
        return Ke[:start_tail]

def electrode_kernel(Karg, start_tail, full_output=False):
    '''
    Extracts the electrode kernel Ke from the raw kernel K
    by removing the membrane kernel, estimated from the
    indexes >= start_tail of the raw kernel.
    full_output = returns Ke,Km if True (otherwise Ke)
    (Ke=electrode filter, Km=membrane filter)
    
    Finds automatically whether to use dendritic or somatic kernel.
    '''

    K = Karg.copy()

    # Fit of the tail to a somatic kernel to find the membrane time constant
    t = arange(len(K))
    tail = arange(start_tail, len(K))
    Ktail = K[tail]
    f = lambda params:params[0] * exp(-params[1] ** 2 * (tail + 1)) - Ktail
    p, _ = optimize.leastsq(f, array([1., .3]))
    Km_soma = p[0] * exp(-p[1] ** 2 * (t + 1))

    f = lambda params:params[0] * ((tail + 1) ** -.5) * exp(-params[1] ** 2 * (tail + 1)) - Ktail
    p, _ = optimize.leastsq(f, array([1., .03]))
    Km_dend = p[0] * ((t + 1) ** -.5) * exp(-p[1] ** 2 * (t + 1))

    if sum((Km_soma[tail] - Ktail) ** 2) < sum((Km_dend[tail] - Ktail) ** 2):
        print "Somatic kernel"
        Km = Km_soma
    else:
        print "Dendritic kernel"
        Km = Km_dend

    K[tail] = Km[tail]

    # Find the minimum
    z = optimize.fminbound(lambda x:sum(solve_convolution(K, x * Km)[tail] ** 2), .5, 1.)
    Ke = solve_convolution(K, z * Km)

    if full_output:
        return Ke[:start_tail], z * Km
    else:
        return Ke[:start_tail]

def AEC_compensate(v, i, ke):
    '''
    Active Electrode Compensation, done offline.
    v = recorded potential
    i = injected current
    ke = electrode kernel
    Returns the compensated potential.
    '''
    return v - convolve(ke, i)[:-(len(ke) - 1)]

def levinson_durbin(a, y):
    '''
    Solves AX=Y where A is a symetrical Toeplitz matrix with coefficients
    given by the vector a (a = first row = first column of A).
    '''
    b = 0 * a
    x = 0 * a
    b[0] = 1. / a[0]
    x[0] = y[0] * b[0]
    for i in range(1, len(a)):
        alpha = sum(a[1:i + 1] * b[:i])
        u = 1. / (1 - alpha ** 2)
        v = -alpha * u
        tmp = b[i - 1]
        if i > 1:
            b[1:i] = v * b[i - 2::-1] + u * b[:i - 1]
        b[0] = v * tmp
        b[i] = u * tmp
        beta = y[i] - sum(a[i:0:-1] * x[:i])
        x += beta * b
    return x
