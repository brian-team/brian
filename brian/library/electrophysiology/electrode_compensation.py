"""
Electrode compensation
"""
from brian.stateupdater import get_linear_equations
from brian.log import log_info
from brian import second, Mohm, mV, ms, Equations, ohm, volt, second
from scipy.optimize import fmin
from scipy.signal import lfilter
from scipy import linalg
from numpy import sqrt, ceil, zeros, eye, poly, dot, hstack, array
import time

__all__=['Lp_compensate'] # more explicit name?

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

def simulate(eqs, I, dt, row=0): # export?
    """
    I must be normalized (I*Re/taue for example)
    """
    M, B = get_linear_equations(eqs)
    A = linalg.expm(M * dt)
    b, a = compute_filter(A, row=row)
    y = lfilter(b, a, I*dt) + B[row]
    return y

class ElectrodeCompensation (object):
    eqs = """
            dV/dt=Re*(-Iinj)/taue : volt
            dV0/dt=(R*Iinj-V0+Vr)/tau : volt
            Iinj=(V-V0)/Re : amp
            """

    def __init__(self, I, Vraw,
                 dt, durslice=1*second,
                  p=1.0, 
                 *params):
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
        R,tau,Vr,Re,taue = x

        R = R*R
        tau = tau*tau
        Re = Re*Re
        taue = taue*taue

        return R,tau,Vr,Re,taue

    def params_to_vector(self, *params):
        x = params

        x = [sqrt(params[0]),
             sqrt(params[1]),
             params[2],
             sqrt(params[3]),
             sqrt(params[4])]

        return list(x)

    def get_model_trace(self, row, *x):
        R, tau, Vr, Re, taue = self.vector_to_params(*x)
        # put units again
        R, tau, Vr, Re, taue = R*ohm, tau*second, Vr*volt, Re*ohm, taue*second
        eqs = Equations(self.eqs)
        eqs.prepare()
        self._eqs = eqs
        y = simulate(eqs, self.I_list[self.islice] * Re/taue, self.dt, row=row)
        return y

    def fitness(self, x):
        R, tau, Vr,  Re, taue = self.vector_to_params(*x)
        y = self.get_model_trace(0, *x)
        e = self.dt_*sum(abs(self.Vraw_list[self.islice]-y)**self.p)
        return e

    def compensate_slice(self, x0):
        fun = lambda x: self.fitness(x)
        x = fmin(fun, x0, maxiter=10000, maxfun=10000)
        return x

    def compensate(self):
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
        self.Vneuron = hstack(Velec_list)
        self.Velec = hstack(Vcomp_list)
        
        return self.Vcomp

def Lp_compensate(I, Vraw, dt, 
               slice_duration=1*second,
               p=1.0,
               full=False,
               **initial_params):
    '''
    * Renvoyer un dictionnaire de parametres avec des vecteurs
    * Option full=True: renvoie trace d'electrode etc
    '''

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

