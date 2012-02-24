'''
Simulation of a cable equation
'''
from brian import *

# Typical equations
eqs=''' # The same equations for the whole neuron, but possibly different parameter values
Im=gl*(El-v)+gNa*m**3*h*(ENa-v) : amp/cm**2 # distributed transmembrane current
gNa : siemens/cm**2 # spatially distributed conductance
dm/dt=(minf-m)/tauinf : 1
# etc
'''

'''
The simulation is in two stages:
1) Solve the discrete system for V(t+dt), where the other variables are constant
and assumed to be known at time t+dt (first iteration: use value at t). This
is a sparse linear system (could be computed with scipy.sparse.linalg?).
2) Calculate Im at t+dt using a backward Euler step with the
values of V from step 1. That is, do one step of implicit Euler.
And possibly repeat (until convergence) (how many times?).

The discretized longitudinal current is:
a/(2*R)*(V[i+1]-2*V[i]+V[i-1])/dx**2

Artificial gridpoints are used so that:
(V[1]-V[-1])/(2*dx) = dV/dt(0)   # =0 in general
(V[N+1]-V[N-1])/(2*dx) = dV/dt(L)   # =0 in general

Consider also the capacitive current (and leak?).

Actually in Mascagni it's more complicated, because we use the
full membrane equation (conductances in front V in particular).
Therefore perhaps we would need sympy to reorganize the equation first
as dV/dt=aV+b (see the Synapses class).

x=scipy.linalg.solve_banded((lower,upper),ab,b)
    lower = number of lower diagonals
    upper = number of upper diagonals
    ab = array(l+u+1,M)
        each row is one diagonal
    a[i,j]=ab[u+i-j,j]
'''
