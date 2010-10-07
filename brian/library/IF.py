'''
Integrate-and-Fire models.
'''

from brian.units import *
from brian.stdunits import *
from brian.membrane_equations import *
from numpy import exp

__all__ = ['leaky_IF', 'perfect_IF', 'exp_IF', 'quadratic_IF', 'Brette_Gerstner', 'Izhikevich', \
         'AdaptiveReset', 'aEIF']

__credits__ = dict(author='Romain Brette (brette@di.ens.fr)',
                 date='April 2008')

#TODO: specific integration methods

"""
******************************************
One-dimensional integrate-and-fire models
******************************************
"""
@check_units(tau=second, El=volt)
def leaky_IF(tau, El):
    '''
    A leaky integrate-and-fire model (membrane equation).
    tau dvm/dt = EL - vm
    '''
    return MembraneEquation(tau) + \
           Current('Im=El-vm:volt', current_name='Im', El=El)

@check_units(tau=second)
def perfect_IF(tau):
    '''
    A perfect integrator.
    tau dvm/dt = ...
    '''
    return MembraneEquation(tau)

@check_units(C=farad, gL=siemens, EL=volt, VT=volt, DeltaT=volt)
def exp_IF(C, gL, EL, VT, DeltaT):
    '''
    An exponential integrate-and-fire model (membrane equation).
    '''
    return MembraneEquation(C) + \
           Current('Im=gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT):amp', \
                   gL=gL, EL=EL, DeltaT=DeltaT, exp=exp, VT=VT)

@check_units(C=farad, a=siemens / volt, EL=volt, VT=volt)
def quadratic_IF(C, a, EL, VT):
    '''
    Quadratic integrate-and-fire model.
    C*dvm/dt=a*(vm-EL)*(vm-VT)
    '''
    return MembraneEquation(C) + \
           Current('Im=a*(vm-EL)*(vm-VT):amp', \
                   a=a, EL=EL, exp=exp, VT=VT)

"""
******************************************
Two-dimensional integrate-and-fire models
******************************************
"""
@check_units(a=1 / second, b=1 / second)
def Izhikevich(a=0.02 / ms, b=0.2 / ms):
    '''
    Returns a membrane equation for the Izhikevich model
    (variables: vm and w).
    '''
    return MembraneEquation(1.) + \
           Current('''
           Im=(0.04/ms/mV)*vm**2+(5/ms)*vm+140*mV/ms-w : volt/second
           dw/dt=a*(b*vm-w)                            : volt/second
           ''', current_name='Im')

@check_units(C=farad, gL=siemens, EL=volt, VT=volt, DeltaT=volt, \
             tauw=second, a=siemens)
def Brette_Gerstner(C=281 * pF, gL=30 * nS, EL= -70.6 * mV, VT= -50.4 * mV, \
                   DeltaT=2 * mV, tauw=144 * ms, a=4 * nS):
    '''
    Returns a membrane equation for the Brette-Gerstner model.
    Default: a regular spiking cortical cell.

    Brette, R. and W. Gerstner (2005).
    Adaptive exponential integrate-and-fire model as an effective
    description of neuronal activity.
    Journal of Neurophysiology 94: 3637-3642.
    '''
    return exp_IF(C, gL, EL, VT, DeltaT) + \
           IonicCurrent('dw/dt=(a*(vm-EL)-w)/tauw:amp', \
                        a=a, EL=EL, tauw=tauw)

aEIF = Brette_Gerstner # synonym
AdEx = aEIF


class AdaptiveReset(object):
    '''
    A two-variable reset:
      V<-Vr
      w<-w+b
    (used in Izhikevich and Brette-Gerstner models)
    '''
    def __init__(self, Vr= -70.6 * mvolt, b=0.0805 * nA):
        self.Vr = Vr
        self.b = b

    def __call__(self, P):
        '''
        Clamps membrane potential at reset value.
        '''
        spikes = P.LS.lastspikes()
        P.vm_[spikes] = self.Vr
        P.w_[spikes] += self.b
