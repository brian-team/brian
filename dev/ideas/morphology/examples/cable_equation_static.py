'''
Simulation of a cable equation
'''
from brian import *
from brian.experimental.morphology import *
from time import time

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

El = 0 * mV
gl = 0.02 * msiemens / cm ** 2

# Typical equations
eqs='''
Im=gl*(El-v)+I : amp/cm**2 # distributed transmembrane current
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, refractory=refractory, Cm=Cm, Ri=Ri)
neuron.v=El
neuron.I=0*amp/cm**2

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
