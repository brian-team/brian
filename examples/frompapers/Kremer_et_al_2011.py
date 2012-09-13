#!/usr/bin/env python
'''
Late Emergence of the Whisker Direction Selectivity Map in the Rat Barrel Cortex
Kremer Y, Leger JF, Goodman DF, Brette R, Bourdieu L (2011). J Neurosci 31(29):10689-700.

Development of direction maps with pinwheels in the barrel cortex.
Whiskers are deflected with random moving bars.
N.B.: network construction can be long.

In this version, STDP is faster than in the paper so that the script runs in just a
few minutes.
'''
from brian import *

# Uncomment if you have a C compiler
# set_global_preferences(useweave=True,usecodegen=True,usecodegenweave=True,usenewpropagate=True,usecstdp=True)

# PARAMETERS
# Neuron numbers
M4,M23exc,M23inh=22,25,12 # side of each barrel (in neurons)
N4,N23exc,N23inh=M4**2,M23exc**2,M23inh**2 # neurons per barrel
barrelarraysize=5 # Choose 3 or 4 if memory error
Nbarrels=barrelarraysize**2
# Stimulation
stim_change_time = 5*ms
Fmax=.5/stim_change_time # maximum firing rate in layer 4 (.5 spike / stimulation)
# Neuron parameters
taum,taue,taui=10*ms,2*ms,25*ms
El=-70*mV
Vt,vt_inc,tauvt=-55*mV,2*mV,50*ms # adaptive threshold
# STDP
taup,taud=5*ms,25*ms
Ap,Ad=.05,-.04
# EPSPs/IPSPs
EPSP,IPSP = 1*mV,-1*mV
EPSC = EPSP * (taue/taum)**(taum/(taue-taum))
IPSC = IPSP * (taui/taum)**(taum/(taui-taum))
# Model: IF with adaptive threshold
eqs='''
dv/dt=(ge+gi+El-v)/taum : volt
dge/dt=-ge/taue : volt
dgi/dt=-gi/taui : volt
dvt/dt=(Vt-vt)/tauvt : volt # adaptation
x : 1
y : 1
'''
# Tuning curve
tuning=lambda theta:clip(cos(theta),0,Inf)*Fmax

# Layer 4
layer4=PoissonGroup(N4*Nbarrels)
barrels4 = dict(((i, j), layer4.subgroup(N4)) for i in xrange(barrelarraysize) for j in xrange(barrelarraysize))
barrels4active = dict((ij, False) for ij in barrels4)
barrelindices = dict((ij, slice(b._origin, b._origin+len(b))) for ij, b in barrels4.iteritems())
layer4.selectivity = zeros(len(layer4))
for (i, j), inds in barrelindices.iteritems():
    layer4.selectivity[inds]=linspace(0,2*pi,N4)

# Layer 2/3
layer23=NeuronGroup(Nbarrels*(N23exc+N23inh),model=eqs,threshold='v>vt',reset='v=El;vt+=vt_inc',refractory=2*ms)
layer23.v=El
layer23.vt=Vt

# Layer 2/3 excitatory
layer23exc=layer23.subgroup(Nbarrels*N23exc)
x,y=meshgrid(arange(M23exc)*1./M23exc,arange(M23exc)*1./M23exc)
x,y=x.flatten(),y.flatten()
barrels23 = dict(((i, j), layer23exc.subgroup(N23exc)) for i in xrange(barrelarraysize) for j in xrange(barrelarraysize))
for i in range(barrelarraysize):
    for j in range(barrelarraysize):
        barrels23[i,j].x=x+i
        barrels23[i,j].y=y+j

# Layer 2/3 inhibitory
layer23inh=layer23.subgroup(Nbarrels*N23inh)
x,y=meshgrid(arange(M23inh)*1./M23inh,arange(M23inh)*1./M23inh)
x,y=x.flatten(),y.flatten()
barrels23inh = dict(((i, j), layer23inh.subgroup(N23inh)) for i in xrange(barrelarraysize) for j in xrange(barrelarraysize))
for i in range(barrelarraysize):
    for j in range(barrelarraysize):
        barrels23inh[i,j].x=x+i
        barrels23inh[i,j].y=y+j

print "Building synapses, please wait..."
# Feedforward connections
feedforward=Connection(layer4,layer23exc,'ge')
for i in range(barrelarraysize):
    for j in range(barrelarraysize):
        feedforward.connect_random(barrels4[i,j],barrels23[i,j],sparseness=.5,weight=EPSC*.5)
stdp=ExponentialSTDP(feedforward,taup,taud,Ap,Ad,wmax=EPSC)

# Excitatory lateral connections
recurrent_exc=Connection(layer23exc,layer23,'ge')
recurrent_exc.connect_random(layer23exc,layer23exc,weight=EPSC*.3,
                             sparseness=lambda i,j:.15*exp(-.5*(((layer23exc.x[i]-layer23exc.x[j])/.4)**2+((layer23exc.y[i]-layer23exc.y[j])/.4)**2)))
recurrent_exc.connect_random(layer23exc,layer23inh,weight=EPSC,
                             sparseness=lambda i,j:.15*exp(-.5*(((layer23exc.x[i]-layer23inh.x[j])/.4)**2+((layer23exc.y[i]-layer23inh.y[j])/.4)**2)))

# Inhibitory lateral connections
recurrent_inh=Connection(layer23inh,layer23exc,'gi')
recurrent_inh.connect_random(layer23inh,layer23exc,weight=IPSC,
                         sparseness=lambda i,j:exp(-.5*(((layer23inh.x[i]-layer23exc.x[j])/.2)**2+((layer23inh.y[i]-layer23exc.y[j])/.2)**2)))

# Stimulation
stimspeed = 1./stim_change_time # speed at which the bar of stimulation moves
direction = 0.0
stimzonecentre = ones(2)*barrelarraysize/2.
stimcentre,stimnorm = zeros(2),zeros(2)
stimradius = (11*stim_change_time*stimspeed+1)*.5
stimradius2 = stimradius**2

def new_direction():
    global direction
    direction = rand()*2*pi
    stimnorm[:] = (cos(direction), sin(direction))
    stimcentre[:] = stimzonecentre-stimnorm*stimradius

@network_operation
def stimulation():
    global direction, stimcentre
    stimcentre += stimspeed*stimnorm*defaultclock.dt
    if sum((stimcentre-stimzonecentre)**2)>stimradius2:
        new_direction()
    for (i, j), b in barrels4.iteritems():
        whiskerpos = array([i,j], dtype=float)+0.5
        isactive = abs(dot(whiskerpos-stimcentre, stimnorm))<.5
        if barrels4active[i, j]!=isactive:
            barrels4active[i, j] = isactive
            b.rate = float(isactive)*tuning(layer4.selectivity[barrelindices[i, j]]-direction)

new_direction()

run(5*second,report='text')

figure()
# Preferred direction
selectivity=array([mean(array(feedforward[:,i].todense())*exp(layer4.selectivity*1j)) for i in range(len(layer23exc))])
selectivity=(arctan2(selectivity.imag,selectivity.real) % (2*pi))*180./pi

I=zeros((barrelarraysize*M23exc,barrelarraysize*M23exc))
ix=array(around(layer23exc.x*M23exc),dtype=int)
iy=array(around(layer23exc.y*M23exc),dtype=int)
I[iy,ix]=selectivity
imshow(I)
hsv()
colorbar()
for i in range(1,barrelarraysize+1):
    plot([i*max(ix)/barrelarraysize,i*max(ix)/barrelarraysize],[0,max(iy)],'k')
    plot([0,max(ix)],[i*max(iy)/barrelarraysize,i*max(iy)/barrelarraysize],'k')

figure()
hist(selectivity)

show()
