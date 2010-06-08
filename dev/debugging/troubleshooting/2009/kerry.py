import numpy, scipy
from numpy import *
from brian import *

#physical characteristics
Cap = 2 * uF / (cm * cm)
GNa = 0.02 * siemens / (cm * cm)
GK = 0.02 * siemens / (cm * cm)
GLeak = 0.002 * siemens / (cm * cm)
ELeak = -70 * mV
EK = -100 * mV
ENa = 50 * mV
#ML characteristics
v1 = -11.2 * mV
v2 = 18 * mV
v3 = -20 * mV
v4 = 10 * mV
phiScalingFactor = 0.15 * mV / mV
TauCofactor = 2 * ms

eqs = Equations('''
dvm/dt=(-GNa*minf*(vm-ENa)-GK*w*(vm-EK)-GLeak(vm-ELeak))/200*pF:mV
minf=0.5*(1+tanh((vm-v1)/v2)):1
winf=0.5*(1+tanh((vm-v3)/v4)):1
TauW=TauCofactor/cosh((vm-v3)/(2*v4)):ms
dw/dt=phiScalingFactor*(winf-w)/TauW:kHz''')
#eqs.compile_functions(True)
neuron = NeuronGroup(1, eqs, freeze=True, method='Euler')
