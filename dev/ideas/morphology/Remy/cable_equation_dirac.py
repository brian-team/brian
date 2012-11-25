if __name__ == '__main__':
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
	from spatialneuron_monoprocess import *
	from time import time

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
	dt = defaultclock.dt

	length=4000*um
	diameter=1*um
	n=300

	Ri=100 * ohm * cm
	Cm=1 * uF / cm ** 2
	refractory=4 * ms

	morpho = Soma(diameter=1 * um)
	morpho.L = Cylinder(length=length/3, diameter=diameter, n=n/3, type='axon')
	morpho.LL = Cylinder(length=length/3, diameter=diameter, n=n/3, type='axon')
	morpho.LLL = Cylinder(length=length/3, diameter=diameter, n=n/3, type='axon')
	
	environment ='''El = 0 * mV
gl = 0.02 * msiemens / cm ** 2
'''
	exec(environment)
	
	# Typical equations
	eqs=''' # The same equations for the whole neuron, but possibly different parameter values
	Im=gl0*(El-v)+I : amp/cm**2 # distributed transmembrane current
	I : amp/cm**2 # applied current
	gl0:siemens/cm**2
	'''

	neuron = SpatialNeuron(morphology=morpho, model=eqs, refractory=refractory, Cm=Cm, Ri=Ri,diffeq_nonzero=False,implicit=False)
	neuron.v=El
	neuron.I=0*amp/cm**2
	neuron.gl= 0.02 * msiemens / cm ** 2
	I0 = 50 * nA/neuron.area[0]
	neuron.I[1]= I0

	M=StateMonitor(neuron,'v',record=True)

	taum = neuron.Cm[0]/gl * second**4 * (volt/ohm)**2 / (metre**4 * kilogram)
	rm = taum/(Cm * pi * diameter)
	ra = (4 * Ri)/(pi * diameter**2)
	la = sqrt(rm/ra)
	dx = length / n

	T = 3*taum
	print 'taum=',taum
	print "lambda = ",la
	print "temps de simulation : ",T/taum," taum"

	start = time()
	run(defaultclock.dt)
	neuron.I = 0*amp/cm**2
	neuron.changed = True
	run(T-defaultclock.dt)
	end = time()
	print end - start," s"

	max = zeros(n)
	for i in range(n):
		for t in range(len(M.times)):
			if M[i][t] > M[i][max[i]]:
				max[i] = t

	i = arange(n)
	p1, = plot(i*dx/mm,max *ms,'r') #t* en fonction de x
	p2, = plot(i*dx/mm, taum / 4 * (sqrt(1+4*(i*dx)**2/la**2)-1) * ms / dt,'g') # t* pour un cable infini
	xlabel('position (mm)')
	ylabel('temps pour lequel le potentiel est maximal')
	legend([p1,p2],["simulation","theorie"], loc='upper left')
	#plot(i*dx/mm, (max - taum / 4 * (sqrt(1+4*(i*dx)**2/la**2)-1) / dt)/max) 

	#for i in range(n-10):
		#plot(M.times/ms,M[35+(i/10)*3]/mV)

	show()
