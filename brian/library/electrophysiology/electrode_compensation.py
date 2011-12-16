"""
Electrode compensation
----------------------
Two methods are implemented:
* Lp parametric compensation
* Active Electrode Compensation (AEC)
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

__all__=['Lp_compensate', 'full_kernel', 'full_kernel_from_step',
         'electrode_kernel_soma', 'electrode_kernel_dendrite', 'solve_convolution',
         'electrode_kernel', 'AEC_compensate']

def compute_filter(A, row=0):
    """
    From a linear discrete-time multidimensional system Y(n+1)=AY(n)+X(n),
    with X(n) a vector with X(n)[1:]==0,
    compute a linear 1D filter for simulating the variable indexed by "row".
    It returns two vectors b, a, that are to be provided to the Scipy
    function lfilter.
    """
    d = len(A)

    # compute a: characteristic polynomial of the matrix
    a = poly(A)  # with a[0]=1

    # compute b recursively
    b = zeros(d+1)
    T = eye(d)
    b[0] = T[row, 0]
    for i in range(1, d+1):
        T = a[i]*eye(d) + dot(A, T)
        b[i] = T[row, 0]

    return b, a

def simulate(eqs, I, dt, row=0): # export?
    """
    Simulate a system of neuron equations in response to an injected current I
    with sampling frequency 1/dt, using a Scipy linear filter rather than a Brian
    integrator (it's more efficient since there are no spikes). 
    The variable indexed by "row" only is simulated.
    I must not appear in the equations, rather, it will be injected in the 
    *first* variable only. That is to say, if the equations correspond to the
    differential system dY/dt=MY+I, where only the first component of I is nonzero,
    then the equations must correspond to the matrix M. 
    For instance, if the equations were dv/dt=(R*I-v)/tau, then the following
    equation must be given: "dv/dt=-v/tau", and the input current will be
    "I*R/tau".
    """
    # get M and B such that the system is dY/dy=M(Y-B)
    # NOTE: it only works if the system is invertible (det(M)!=0) !
    M, B = get_linear_equations(eqs)
    # discretization of the system: Y(n+1)=AY(n), were A=exp(M*dt).
    A = linalg.expm(M * dt)
    # compute the filter
    b, a = compute_filter(A, row=row)
    # apply the filter to the injected current
    y = lfilter(b, a, I*dt) + B[row]
    return y

"""
Lp Electrode Compensation
-------------------------
From: Rossant et al (?): to be submitted.
"""
class ElectrodeCompensation (object):
    """
    Lp Electrode Compensation class.
    ================================

    Implements an electrode compensation technique based on linear model 
    fitting of an electrode and a neuron in response to an injected current.
    The fitness function is the Lp error between the full model response
    and the raw trace, with p<2 to minimize the bias due to the nonlinear
    voltage excursions of action potentials.
    The fitting procedure uses the Scipy fmin optimization function.
    The model simulation is performed efficiently on the CPU using a
    linear filter directly applied to the current. This filter is
    computed dynamically from the electrode and neuron differential equations.
    """
    eqs = """
            dV/dt=Re*(-Iinj)/taue : volt
            dV0/dt=(R*Iinj-V0+Vr)/tau : volt
            Iinj=(V-V0)/Re : amp
            """

    def __init__(self, I, Vraw,
                 dt, durslice=10*second,
                  p=1.0, 
                 *params):
        """
        Class constructor.

        * I: injected current, 1D vector.
        * Vraw: raw (uncompensated) voltage trace, 1D vector, same length as I.
        * dt: sampling period (inverse of the sampling frequency), in second.
        * durslice=1*second: duration of slices: the fit is performed independently on each slice
          to capture the possible parameter changes during the recordings.
        * p=1.0: parameter of the Lp error. In general, p<2, and a 
          smaller value for p (like 0.5) will yield better results, especially if
          there are a lot of action potentials.
        * *params: a list of initial parameters for the optimization, in the following
          order: R, tau, Vr, Re, taue.
        """
        self.I = I
        self.Vraw = Vraw
        self.p = p
        self.dt = dt
        self.dt_ = float(dt)
        self.x0 = self.params_to_vector(*params)
        self.duration = len(I) * dt
        self.durslice = min(durslice, self.duration)
        self.slicesteps = int(durslice/dt)
        self.nslices = int(ceil(len(I)*dt/durslice))
        
        self.islice = 0
        self.I_list = [I[self.slicesteps*i:self.slicesteps*(i+1)] for i in range(self.nslices)]
        self.Vraw_list = [Vraw[self.slicesteps*i:self.slicesteps*(i+1)] for i in range(self.nslices)]

    def vector_to_params(self, *x):
        """
        Convert a vector of parameters (used for the optimization) to a tuple
        of actual parameters.
        """
        R,tau,Vr,Re,taue = x

        R = R*R
        tau = tau*tau
        Re = Re*Re
        taue = taue*taue

        return R,tau,Vr,Re,taue

    def params_to_vector(self, *params):
        """
        Opposite of vector_to_params.
        """
        x = params

        x = [sqrt(params[0]),
             sqrt(params[1]),
             params[2],
             sqrt(params[3]),
             sqrt(params[4])]

        return list(x)

    def get_model_trace(self, row, *x):
        """
        Compute the model response (variable index "row") to the injected current,
        at a specific slice (stored in self.islice), with model parameters
        specified with the vector x.
        """
        R, tau, Vr, Re, taue = self.vector_to_params(*x)
        # put units again
        R, tau, Vr, Re, taue = R*ohm, tau*second, Vr*volt, Re*ohm, taue*second
        eqs = Equations(self.eqs)
        eqs.prepare()
        self._eqs = eqs
        y = simulate(eqs, self.I_list[self.islice] * Re/taue, self.dt, row=row)
        return y

    def fitness(self, x):
        """
        fitness function provided to the fmin optimization procedure.
        Simulate the model and compute the Lp error between the model
        response and the raw trace.
        """
        R, tau, Vr,  Re, taue = self.vector_to_params(*x)
        y = self.get_model_trace(0, *x)
        e = self.dt_*sum(abs(self.Vraw_list[self.islice]-y)**self.p)
        return e

    def compensate_slice(self, x0):
        """
        Compensate on the current slice, by calling fmin on the fitness function.
        """
        fun = lambda x: self.fitness(x)
        x = fmin(fun, x0, maxiter=10000, maxfun=10000, disp=False)
        return x

    def compensate(self):
        """
        Compute compensate_slice for all slices.
        Use the previous best parameters as initial parameters for the next slice.
        """
        self.params_list = []
        self.xlist = [self.x0]
        t0 = time.clock()
        for self.islice in range(self.nslices):
            newx = self.compensate_slice(self.xlist[self.islice])
            self.xlist.append(newx)
            self.params_list.append(self.vector_to_params(*newx))
            log_info("electrode_compensation","Slice %d/%d compensated in %.2f seconds" %  \
                (self.islice+1, self.nslices, time.clock()-t0))
            t0 = time.clock()
        self.xlist = self.xlist[1:]
        return self.xlist

    def get_compensated_trace(self):
        """
        Called after the compensation procedure. Compute the full model
        traces, for all slices, using the best parameters found by the
        optimization. Returns only the compensated trace, but this
        function also computes the neuron and electrode responses.
        """
        Vcomp_list = []
        Vneuron_list = []
        Velec_list = []
        
        for self.islice in range(self.nslices):
            x = self.xlist[self.islice]
            V = self.get_model_trace(0, *x)
            V0 = self.get_model_trace(1, *x)
            Velec = V-V0
            
            Vneuron_list.append(V0)
            Velec_list.append(Velec)
            Vcomp_list.append(self.Vraw_list[self.islice] - Velec)
            
        self.Vcomp = hstack(Vcomp_list)
        self.Vneuron = hstack(Vneuron_list)
        self.Velec = hstack(Velec_list)
        
        return self.Vcomp

def Lp_compensate(I, Vraw, dt, 
               slice_duration=1*second,
               p=1.0,
               full=False,
               **initial_params):
    """
    Lp Electrode Compensation.

    Implements an electrode compensation technique based on linear model 
    fitting of an electrode and a neuron in response to an injected current.
    The fitness function is the Lp error between the full model response
    and the raw trace, with p<2 to minimize the bias due to the nonlinear
    voltage excursions of action potentials.
    The fitting procedure uses the Scipy fmin optimization function.
    The model simulation is performed efficiently on the CPU using a
    linear filter directly applied to the current. This filter is
    computed dynamically from the electrode and neuron differential equations.

    Arguments
    * I: injected current, 1D vector.
    * Vraw: raw (uncompensated) voltage trace, 1D vector, same length as I.
    * dt: sampling period (inverse of the sampling frequency), in second.
    * slice_duration=1*second: duration of slices: the fit is performed independently on each slice
        to capture the possible parameter changes during the recordings.
    * p=1.0: parameter of the Lp error. In general, p<2, and a 
        smaller value for p (like 0.5) will yield better results, especially if
        there are a lot of action potentials.
    * **initial_params: initial parameters for the optimization: R, tau, Vr, Re, taue.
        * R: neuron resistance, default 100 MOhm
        * tau: neuron membrane time constant, default 20 ms
        * Vr: neuron rest potentiel, default -70 mV
        * Re: electrode resistance, default 50 MOhm
        * taue: electrode time constant, default 0.5 ms
    """
    R = initial_params.get("R", 100*Mohm)
    tau = initial_params.get("tau", 20*ms)
    Vr = initial_params.get("Vr", -70*mV)
    Re = initial_params.get("Re", 50*Mohm)
    taue = initial_params.get("taue", .5*ms)

    comp = ElectrodeCompensation(I, Vraw,
                                 dt,
                                 slice_duration,
                                 p,
                                 R, tau, Vr, Re, taue)
    comp.compensate()
    Vcomp = comp.get_compensated_trace()
    params = array(comp.params_list).transpose()
    if not full:
        return Vcomp, params
    else:
        return dict(Vcompensated=Vcomp, Vneuron=comp.Vneuron,
                    Velectrode=comp.Velec, params=params)

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
