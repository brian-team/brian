'''
Simulation of a cable equation

Seems to work (matches examples/misc/cable).

TODO:
* Add active currents (state updater)
* Check algorithm (e.g. w/ Hines)
* Add point processes
* Add branches
'''
from brian import *
from morphology import *
from spatialneuron import *

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

defaultclock.dt=0.1*ms

morpho=Cylinder(length=1000*um, diameter=1*um, n=100, type='axon')

El = 0 * mV
gl = 0.02 * msiemens / cm ** 2

# Typical equations
eqs=''' # The same equations for the whole neuron, but possibly different parameter values
Im=gl*(El-v)+I : amp/cm**2 # distributed transmembrane current
I : amp/cm**2 # applied current
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, refractory=4 * ms, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
neuron.v=El
neuron.I=0*amp/cm**2
neuron.I[50]=.05 * nA/neuron.area[0]
M=StateMonitor(neuron,'v',record=True)

print 'taum=',neuron.Cm/gl

run(1*ms)
neuron.I=0
run(200*ms)

for i in range(10):
    plot(M.times/ms,M[35+i*3]/mV)
show()
