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
	from spatialneuron import *
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
	
	defaultclock.dt=1*ms
	
	n0 = 300
	n1 = 200
	n2 = 700
	n=n0+n1+n2

	L0 = n0*2*um
	L1 = n1*2*um
	L2 = n2*2*um
	d0 = 1*um
	d1 = 0.8*um
	d2 = 1.3*um
	
	Ri=100 * ohm * cm
	Cm=1 * uF / cm ** 2
	refractory=4 * ms
	dx0 = L0 / n0
	
	morpho = Cylinder(length=dx0, diameter=d0, n=1, type='axon')
	morpho.L = Cylinder(length=L0, diameter=d0, n=n0, type='axon')
	morpho.LL = Cylinder(length=L1, diameter=d1, n=n1, type='axon')
	morpho.LR = Cylinder(length=L2, diameter=d2, n=n2, type='axon')
	
	environment ='''El = 0 * mV
gl = 0.02 * msiemens / cm ** 2
	'''
	exec(environment)
	
	# Typical equations
	eqs=''' # The same equations for the whole neuron, but possibly different parameter values
	Im=gl*(El-v)+I : amp/cm**2 # distributed transmembrane current
	I : amp/cm**2 # applied current
	'''
	
	neuron = SpatialNeuron(morphology=morpho, model=eqs, refractory=refractory, Cm=Cm, Ri=Ri, environment=environment,diffeq_nonzero=False)
	neuron.v=El
	# neuron.I=0*amp/cm**2
	# I0 = .05 * nA/neuron.area[1]
	# neuron.I[1]= I0
	
	M=StateMonitor(neuron,'v',record=True)
	
	taum = neuron.Cm[0]/gl * second**4 * (volt/ohm)**2 / (metre**4 * kilogram)
	
	rm0 = taum/(Cm * pi * d0)
	ra0 = (4 * Ri)/(pi * d0**2)
	la0 = sqrt(rm0/ra0)
	dx0 = L0 / n0
	CL0 = cosh(L0 / la0)
	SL0 = sinh(L0 / la0)
	R_infinity_0 = sqrt(rm0 * ra0)
	
	rm1 = taum/(Cm * pi * d1)
	ra1 = (4 * Ri)/(pi * d1**2)
	la1 = sqrt(rm1/ra1)
	dx1 = L1 / n1
	CL1 = cosh(L1 / la1)
	SL1 = sinh(L1 / la1)
	R_infinity_1 = sqrt(rm1 * ra1)
	R_in_1 = R_infinity_1 * CL1/SL1
	
	rm2 = taum/(Cm * pi * d2)
	ra2 = (4 * Ri)/(pi * d2**2)
	la2 = sqrt(rm2/ra2)
	dx2 = L2 / n2
	CL2 = cosh(L2 / la2)
	SL2 = sinh(L2 / la2)
	R_infinity_2 = sqrt(rm2 * ra2)
	R_in_2 = R_infinity_2 * CL2/SL2
	
	R_L_0 = (R_in_2 * R_in_1)/(R_in_2 + R_in_1)
	R_in_0 = R_infinity_0 * (R_L_0 + R_infinity_0 * SL0/CL0)/(R_infinity_0 + R_L_0 * SL0/CL0)
	
	# V0_th = R_in_0 * I0 * neuron.area[1]
	
	V0_th = 100 * mV
	neuron.v[0] = V0_th
	neuron.bc[0] = 0
	T = 30*taum
	
	print 'taum=',taum
	print "temps de simulation : ",T/taum," taum"
	
	start = time()
	run(T)
	end = time()
	
	print end - start," s"
	
	testTime = (T-1*ms)/defaultclock.dt
	
	i = arange(n+1)
	i0 = arange(n0+1)
	i1 = arange(n0,n0+n1+1)
	i2 = arange(n0+n1,n0+n1+n2+1)
	photo = zeros(n+1)
	for k in xrange(n+1):
		photo[k] = M[k][testTime]
	p1, = plot (i ,photo/mV,'r')
	
	p2, = plot(i0,V0_th * (R_L_0 * cosh((L0-i0*dx0)/la0) + R_infinity_0 * sinh((L0-i0*dx0)/la0))/(R_L_0 * CL0 + R_infinity_0 * SL0) / mV,'g')
	V1 = V0_th * (R_L_0 * cosh((L0-n0*dx0)/la0) + R_infinity_0 * sinh((L0-n0*dx0)/la0))/(R_L_0 * CL0 + R_infinity_0 * SL0)
	p3, = plot(i1, V1 / CL1 * cosh((L1 - (i1 - n0) * dx1) /  la1 ) / mV,'g')
	p4, = plot(i2, V1 / CL2 * cosh((L2 - (i2 - (n0+n1)) * dx2) /  la2 ) / mV,'g')
	xlabel('position (compartiment)')
	ylabel('potentiel (mV)')
	legend([p1,p2],["simulation","theorie"], loc='upper right')
	"""
	for i in range(n):
		#plot(M.times/ms,M[35+(i/10)*3]/mV)	
	"""
	show()
