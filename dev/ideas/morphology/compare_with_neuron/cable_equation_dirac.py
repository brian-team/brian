# It works!
from brian import *
from brian.experimental.morphology import *
from time import time

defaultclock.dt=0.1*ms
dt = defaultclock.dt

length=4000*um
diameter=1*um
n=300

Ri=100 * ohm * cm
Cm=1 * uF / cm ** 2
refractory=4 * ms

morpho = Soma(diameter=1 * um)
morpho.length=0.1*um
morpho.L = Cylinder(length=length/3, diameter=diameter, n=n/3, type='axon')
morpho.LL = Cylinder(length=length/3, diameter=diameter, n=n/3, type='axon')
morpho.LLL = Cylinder(length=length/3, diameter=diameter, n=n/3, type='axon')

El = 0 * mV
gl = 0.02 * msiemens / cm ** 2

# Typical equations
eqs=''' # The same equations for the whole neuron, but possibly different parameter values
Im=gl*(El-v)+I : amp/cm**2 # distributed transmembrane current
I : amp/cm**2 # applied current
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, refractory=refractory, Cm=Cm, Ri=Ri)
neuron.v=El
neuron.I=0*amp/cm**2
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
