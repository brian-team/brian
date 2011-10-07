from brian import *
from numpy import random
from scipy import sparse,linalg

N_PY = 100
alpha_x = 1.0

t_final = 0.01*second

sim_clock = Clock(dt=0.05*ms)

PY = NeuronGroup(N=N_PY,
                 model='V:1',clock=sim_clock,
                 #threshold=EmpiricalThreshold(state='Vs',threshold=10,refractory=3*ms,clock=sim_clock),
                 threshold=10,
                 order=2,freeze=True)

CPP = zeros((N_PY,N_PY))
CPP[20,50] = 1.

CPP = sparse.lil_matrix(CPP)

C = Connection(PY,PY,'V',weight=alpha_x*CPP)

run(t_final)
