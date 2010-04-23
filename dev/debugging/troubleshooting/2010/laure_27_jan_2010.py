# -*- coding:utf-8 -*-
from brian import *
from brian.library.IF import *
import numpy

# Parameters
area=20000*umetre**2
Cm=(20*ufarad*cm**-2)*area
# Time constants
taue=5*ms
taui=10*ms
#Maximal conductances
gK = (8.0*usiemens*cm**-2)*area
gCa =(4.4*usiemens*cm**-2)*area #type II
gL = (2.0*usiemens*cm**-2)*area
# Reversal potentials
EK = -80*mV
ECa =-60*mV
EL = 120*mV
Ee=1*mV#0*mV
Ei=-80*mV
v1 = -1.2*mV
v2 = 18*mV
v3 = 2*mV
v4 = 30*mV

we=30*nS ##6# excitatory synaptic weight (voltage)
wi=67*nS # inhibitory synaptic weight

phi = 0,04*1 # that's the typo: should be 0.04*1
Vr=-70*mV
threshold=48*mV#-30

# Model
##Morris-Lecar##
eqsRS = Equations('''
dvm/dt=(gL*(EL-vm)+ gK*w/mV*(EK-vm)+ gCa*minf*(ECa-vm) +(ge*(Ee-vm)+gi*(Ei-vm)))/Cm + I:volt
dw/dt= phi*(winf*mV-w)/tauw: volt
minf = 0.5*(1+exp(2*((vm-v1)/v2))-1)/(exp(2*((vm-v1)/v2))+1):1
winf = 0.5*(1+exp(2*((vm-v3)/v4))-1)/(exp(2*((vm-v3)/v4))+1):1
tauw = 1*ms/((exp((vm-v3)/v4)+exp(-(vm-v3)/v4))/2):second
dge/dt = -ge*(1./taue): siemens
dgi/dt = -gi*(1./taui): siemens
I: volt/second
''')
group = NeuronGroup(2, model=eqsRS,method='Euler')
