'''
Synapse models for Brian (no plasticity here).
'''
from brian.equations import *
from brian.units import check_units, second, amp, siemens
from brian.membrane_equations import Current

__all__ = ['exp_current', 'alpha_current', 'biexp_current', \
         'exp_conductance', 'alpha_conductance', 'biexp_conductance', \
         'exp_synapse', 'alpha_synapse', 'biexp_synapse']

__credits__ = dict(author='Romain Brette (brette@di.ens.fr)',
                 date='April 2008')

# -----------------
# Synaptic currents
# -----------------
@check_units(tau=second)
def exp_current(input, tau, current_name=None, unit=amp):
    '''
    Exponential synaptic current.
    input = name of input variable (where presynaptic spikes act).
    current_name = name of current variable
    '''
    current_name = current_name or unique_id()
    current = Current() + exp_synapse(input, tau, unit, current_name)
    current.set_current_name(current_name)
    return current

@check_units(tau=second)
def alpha_current(input, tau, current_name=None, unit=amp):
    '''
    Alpha synaptic current.
    current_name = name of current variable
    '''
    current_name = current_name or unique_id()
    current = Current() + alpha_synapse(input, tau, unit, current_name)
    current.set_current_name(current_name)
    return current

@check_units(tau1=second, tau2=second)
def biexp_current(input, tau1, tau2, current_name=None, unit=amp):
    '''
    Biexponential synaptic current.
    current_name = name of current variable
    '''
    current_name = current_name or unique_id()
    current = Current() + biexp_synapse(input, tau1, tau2, unit, current_name)
    current.set_current_name(current_name)
    return current

# ---------------------
# Synaptic conductances
# ---------------------
@check_units(tau=second)
def exp_conductance(input, E, tau, conductance_name=None, unit=siemens):
    '''
    Exponential synaptic conductance.
    conductance_name = name of conductance variable
    E = synaptic reversal potential
    '''
    conductance_name = conductance_name or unique_id()
    return Current('I=g*(E-vm): amp', I=input + '_current', g=conductance_name, E=E) + \
           exp_synapse(input, tau, unit, conductance_name)

@check_units(tau=second)
def alpha_conductance(input, E, tau, conductance_name=None, unit=siemens):
    '''
    Alpha synaptic conductance.
    conductance_name = name of conductance variable
    E = synaptic reversal potential
    '''
    conductance_name = conductance_name or unique_id()
    return Current('I=g*(E-vm): amp', I=input + '_current', g=conductance_name, E=E) + \
           alpha_synapse(input, tau, unit, conductance_name)

@check_units(tau1=second, tau2=second)
def biexp_conductance(input, E, tau1, tau2, conductance_name=None, unit=siemens):
    '''
    Exponential synaptic conductance.
    conductance_name = name of conductance variable
    E = synaptic reversal potential
    '''
    conductance_name = conductance_name or unique_id()
    return Current('I=g*(E-vm): amp', I=input + '_current', g=conductance_name, E=E) + \
           biexp_synapse(input, tau1, tau2, unit, conductance_name)

# ---------------
# Synaptic inputs
# ---------------
@check_units(tau=second)
def exp_synapse(input, tau, unit, output=None):
    '''
    Exponentially decaying synaptic current/conductance:
    g(t)=exp(-t/tau)
    output = output variable name (plugged into the membrane equation).
    input = input variable name (where spikes are received).
    '''
    if output is None:
        output = input + '_out'
    return Equations('''
    dx/dt = -x*invtau    : unit
    y=x''', x=output, y=input, unit=unit, invtau=1. / tau)

@check_units(tau=second)
def alpha_synapse(input, tau, unit, output=None):
    '''
    Alpha synaptic current/conductance:
    g(t)=(t/tau)*exp(1-t/tau)
    output = output variable name (plugged into the membrane equation).
    input = input variable name (where spikes are received).
    The peak is 1 at time t=tau.
    '''
    if output is None:
        output = input + '_out'
    return Equations('''
    dx/dt = (y-x)*invtau : unit 
    dy/dt = -y*invtau    : unit
    ''', x=output, y=input, unit=unit, invtau=1. / tau)

@check_units(tau1=second, tau2=second)
def biexp_synapse(input, tau1, tau2, unit, output=None):
    '''
    Biexponential synaptic current/conductance:
    g(t)=(tau2/(tau2-tau1))*(exp(-t/tau1)-exp(-t/tau2))
    output = output variable name (plugged into the membrane equation).
    input = input variable name (where spikes are received).
    The peak is 1 at time t=tau1*tau2/(tau2-tau1)*log(tau2/tau1) 
    '''
    if output is None:
        output = input + '_out'
    invpeak = (tau2 / tau1) ** (tau1 / (tau2 - tau1))
    return Equations('''
    dx/dt = (invpeak*y-x)*invtau1 : unit
    dy/dt = -y*invtau2            : unit
    ''', x=output, y=input, unit=unit, invtau1=1. / tau1, invtau2=1. / tau2, invpeak=invpeak)
