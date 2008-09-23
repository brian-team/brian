'''
Ionic currents for Brian
'''
from brian.units import check_units,siemens,volt
from brian.membrane_equations import Current

__credits__=dict(author    = 'Romain Brette (brette@di.ens.fr)',
                 date      = 'April 2008')

@check_units(El=volt)
def leak_current(gl,El,current_name=None):
    '''
    Leak current: gl*(El-vm)
    '''
    return Current('I=gl*(El-vm) : amp',gl=gl,El=El,I=current_name)

#check_units(EK=volt)
def K_current_HH(gmax,EK,current_name=None):
    '''
    Hodkin-Huxley K+ current.
    Resting potential is 0 mV.
    '''
    return Current('''
    I=gmax*n**4*(EK-vm) : amp
    dn/dt=alphan*(1-n)-betan*n : 1
    alphan=.01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
    betan=.125*exp(-.0125*vm/mV)/ms : Hz
    ''',gmax=gmax,EK=EK,I=current_name)

#check_units(ENa=volt)
def Na_current_HH(gmax,ENa,current_name=None):
    '''
    Hodkin-Huxley Na+ current.
    '''
    return Current('''
    I=gmax*m**3*h*(ENa-vm) : amp
    dm/dt=alpham*(1-m)-betam*m : 1
    dh/dt=alphah*(1-h)-betah*h : 1
    alpham=.1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
    betam=4*exp(-.0556*vm/mV)/ms : Hz
    alphah=.07*exp(-.05*vm/mV)/ms : Hz
    betah=1./(1+exp(3.-.1*vm/mV))/ms : Hz
    ''',gmax=gmax,ENa=ENa,I=current_name)
