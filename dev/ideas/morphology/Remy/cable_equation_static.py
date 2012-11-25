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

	length=3000*um
	diameter=1*um
	n = 100
	
	br = 0 #(+1)
	
	Ri=100 * ohm * cm
	Cm=1 * uF / cm ** 2
	refractory=4 * ms
	
	length = (length/(br+1))* (br+1)
	n = (n/(br+1)) * (br+1)
	dx = length / n

	morpho = Cylinder(length=dx, diameter=diameter, n=1, type='axon')
	morpho.L = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')
	if br>0: morpho.LL = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')
	if br>1: morpho.LLL = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')
	if br>2: morpho.LLLL = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')
	if br>3: morpho.LLLLL = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')
	if br>4: morpho.LLLLLL = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')
	if br>5: morpho.LLLLLLL = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')
	if br>6: morpho.LLLLLLLL = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')
	if br>7: morpho.LLLLLLLLL = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')
	if br>8: morpho.LLLLLLLLLL = Cylinder(length=length/(br+1), diameter=diameter, n=n/(br+1), type='axon')

	environment ='''El = 0 * mV
gl = 0.02 * msiemens / cm ** 2
'''
	exec(environment)

	# Typical equations
	eqs=''' # The same equations for the whole neuron, but possibly different parameter values
	Im=gl*(El-v)+I : amp/cm**2 # distributed transmembrane current
	I : amp/cm**2 # applied current
	'''

	neuron = SpatialNeuron(morphology=morpho, model=eqs, refractory=refractory, Cm=Cm, Ri=Ri,diffeq_nonzero=False)
	neuron.v=El
	neuron.I=0*amp/cm**2
	neuron.bc[0] = 0

	# M=StateMonitor(neuron,'v',record=True)

	taum = neuron.Cm[0]/gl * second**4 * (volt/ohm)**2 / (metre**4 * kilogram)
	rm = taum/(Cm * pi * diameter)
	ra = (4 * Ri)/(pi * diameter**2)
	la = sqrt(rm/ra)

	CL = cosh(length / la)
	SL = sinh(length / la)
	R_infinity = sqrt(rm * ra)
	R_sealed = R_infinity * CL/SL
	# V0_lim_sealed = R_sealed * I0 * neuron.area[1]
	R_killed = R_infinity * SL/CL
	# V0_lim_killed = R_killed * I0 * neuron.area[1]

	V0_lim_sealed = 100 * mV
	neuron.v[0] = V0_lim_sealed
	T = 20*taum

	print 'taum=',taum
	print "lambda = ",la
	print "temps de simulation : ",T/taum," taum"
	print "longueur : ",length/la," lambda"

	start = time()
	run(T)
	end = time()

	print end - start," s"

	#testTime = (T-1*ms)/defaultclock.dt
	
	i = arange(n+1)
	photo = zeros(n+1)
	diff = 0
	k_max = 0
	for k in xrange(n+1):
		# photo[k] = M[k][testTime]
		photo[k] = neuron.v[k]
		if k>0 : 
			if diff < abs(photo[k]/mV * metre**2 * kilogram /(second**3 * volt/ohm)- V0_lim_sealed / CL * cosh((length - k * dx) /  la ) / mV) :
				diff = abs(photo[k]/mV * metre**2 * kilogram /(second**3 * volt/ohm)- V0_lim_sealed / CL * cosh((length - k * dx) /  la ) / mV)
				k_max = k
	print diff
	print k_max
	plot (i * dx/mm,photo/mV,'r')
	plot(i*dx / mm,V0_lim_sealed / CL * cosh((length - i * dx) /  la ) / mV,'g')
	xlabel('position (mm)')
	ylabel('potentiel (mV)')
	# plot(i*dx / mm,V0_lim_killed / SL * sinh((length - i * dx) /  la ) / mV,'b')
	
	# for i in range(n):
		# plot(M.times/ms,M[i]/mV)	
	
	show()
