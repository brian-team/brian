'''
Hodgkin-Huxley equations (1952)
'''
if __name__ == '__main__':
	import brian_no_units
	from brian import *
	from morphology import *
	from spatialneuron import *
	#from spatialneuron_monoprocess import *
	#from spatialneuron_float import *
	from time import time
	from numpy import zeros
	
	defaultclock.dt=0.1*ms
	
	length = 10*cm
	diameter=2*238*um
	compartments=1024

	morpho = Soma(diameter=1 * um)
	morpho.L = Cylinder(length=length, diameter=diameter, n=compartments, type='axon')
	"""
	morpho.L = Cylinder(length=length/5, diameter=diameter, n=compartments/5, type='axon')
	morpho.LL = Cylinder(length=length/5, diameter=diameter, n=compartments/5, type='axon')
	morpho.LLL = Cylinder(length=length/5, diameter=diameter, n=compartments/5, type='axon')
	morpho.LLLL = Cylinder(length=length/5, diameter=diameter, n=compartments/5, type='axon')
	morpho.LLLLL = Cylinder(length=length/5, diameter=diameter, n=compartments/5, type='axon')
	#"""
	#morpho = Morphology('mp_ma_40984_gc2.CNG.swc')
	#morpho = Morphology('oi24rpy1.CNG.swc')
	
	
	El = 10.613* mV
	ENa = 115*mV
	EK = -12 * mV
	gl = 0.3 * msiemens / cm ** 2
	gNa = 120 * msiemens / cm ** 2
	gK = 36 * msiemens / cm ** 2
	
	"""
	El = 0 * mV
	gl = 0.02 * msiemens / cm ** 2
	"""
	
	# Typical equations
	
	eqs=''' # The same equations for the whole neuron, but possibly different parameter values
	Im=gl*(El-v)+gNa*m**3*h*(ENa-v)+gK*n**4*(EK-v)+I : amp/cm**2 # distributed transmembrane current
	I:amp/cm**2 # applied current
	dm/dt=alpham*(1-m)-betam*m : 1
	dn/dt=alphan*(1-n)-betan*n : 1
	dh/dt=alphah*(1-h)-betah*h : 1
	alpham=(0.1/mV)*(-v+25*mV)/(exp((-v+25*mV)/(10*mV))-1)/ms : Hz
	betam=4.*exp(-v/(18*mV))/ms : Hz
	alphah=0.07*exp(-v/(20*mV))/ms : Hz
	betah=1./(exp((-v+30*mV)/(10*mV))+1)/ms : Hz
	alphan=(0.01/mV)*(-v+10*mV)/(exp((-v+10*mV)/(10*mV))-1)/ms : Hz
	betan=0.125*exp(-v/(80*mV))/ms : Hz
	'''
	
	"""
	eqs=''' # The same equations for the whole neuron, but possibly different parameter values
	Im=gl*(El-v)+I : amp/cm**2 # distributed transmembrane current
	I : amp/cm**2 # applied current
	'''
	"""
	neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=35.4 * ohm * cm)
	#neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=35.4 * ohm * cm,diffeq_nonzero=False)
	neuron.v=0*mV
	neuron.h=1
	neuron.m=0
	neuron.n=0
	neuron.I=0*amp/cm**2
	M=StateMonitor(neuron,'v',record=True)
	
	#print 'taum=',neuron.Cm[0]/gl * second**4 * (volt/ohm)**2 / (metre**4 * kilogram)
	
	insert_point = 1
	start = time()
	"""
	run(30*ms)
	neuron.I[insert_point] = .07 * uA/neuron.area[insert_point]
	neuron.changed = True
	run(0.04*ms)
	neuron.I=0*amp/cm**2
	neuron.changed = True
	run(0.01*ms)
	morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	run(0.1*ms)
	morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	run(0.3*ms)
	morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	run(0.5*ms)
	morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	run(0.5*ms)
	morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	run(1*ms)
	morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	run(0.3*ms)
	morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	run(0.3*ms)
	morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	#run(1*ms)
	#morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	#run(0.3*ms)
	#morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	#run(5*ms)
	#morpho.plot(simple=True,values=neuron.v,max_val=50*mV,min_val=-10*mV)
	#"""
	"""
	run(0.01*ms)
	#"""
	#"""
	run(30*ms)
	neuron.I[insert_point] = .7 * uA/neuron.area[insert_point]
	neuron.changed = True
	run(3*ms)
	neuron.I[insert_point] = 0 *amp/cm**2
	neuron.changed = True
	run(50*ms)
	#"""
	"""
	run(0.01*ms)
	start = time()
	run(50*ms)
	#"""
		
	end = time()
	
	print end - start," s"
	
	#"""
	p = []
	for i in range(compartments/50):
		pp, = plot(M.times/ms,M[i*50+1]/mV)
		p.append(pp)
	xlabel('temps (ms)')
	ylabel('potentiel (mV)')
	legend(p,["V au compartiment 1","V au compartiment 51","V au compartiment 101","V au compartiment 151","V au compartiment 201"])
	#"""
	
	"""
	p1, = plot(M.times/ms,neuron._state_updater.timeUpdater[1:],'r')
	p2, = plot(M.times/ms,neuron._state_updater.timeDevice[1:],'b')
	p3, = plot(M.times/ms,neuron._state_updater.timeDeviceU[1:],'b--')
	p4, = plot(M.times/ms,neuron._state_updater.timeDeviceT[1:],'b-.')
	p5, = plot(M.times/ms,neuron._state_updater.timeHost[1:],'g')
	p6, = plot(M.times/ms,neuron._state_updater.timeSolveHost[1:],'g--')
	p7, = plot(M.times/ms,neuron._state_updater.timeFillHost[1:],'g-.')
	p8, = plot(M.times/ms,neuron._state_updater.timeFin[1:],'g:')
	legend([p1,p2,p3,p4,p5,p6,p7,p8],["Courant","Branches","DU","DT","Points de jonction","SH","FH","Fin"], loc='upper left')
	#"""
	show()