from brian import *
from brian.library.ionic_currents import *

eqs = MembraneEquation(C=0.2 * nF)
eqs += leak_current(gl=10 * nS, El= -70 * mV)
eqs += K_current_HH(20 * nS, -80 * mV)
print eqs
