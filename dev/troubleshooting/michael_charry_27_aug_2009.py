from brian import *
eqw=Current('I=(vm-V0)/R : amp')+MembraneEquation(200*pF)
print eqw
