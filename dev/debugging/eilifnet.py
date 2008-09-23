# Created by Eugene M. Izhikevich, 2003 Modified by S. Fusi 2007
# Ported to Python by Eilif Muller, 2008.
#import brian_no_units
from brian import *
from numpy.random import uniform

# For measuring performance
import time
t1 = time.time()

# Excitatory and inhibitory neuron counts
Ne = 1000      
Ni = 4
N = Ne+Ni

# Synaptic couplings
Je = 250.0/Ne * mV
Ji = 0.0 * mV

# reset depolarization (mV)
reset = 0.0 * mV
threshold = 20*mV

# refractory period (ms)
refr = 2.5 * ms

# (mV/ms) (lambda is a python keyword)
leak = 5.0*mV/ms
myclock=Clock(0.05*ms)

# Statistics of the background external current
mb = 3.0*mV/ms
sb = 4.0*mV/ms
mue = mb
sigmae=sb
sigmai = 0.0*mV/ms

tau_i=1*ms
dv=lambda v,ie:((v>reset)|((v<=reset) & (-leak+ie>0*mV/ms)))*(-leak+ie+(0./ms/mV)*v**2)
die=lambda v,ie:(mue-ie)/tau_i+sigmae*(.5*tau_i)**(-.5)*xi
die2=lambda v,ie:(mue-ie)/tau_i
Pe=NeuronGroup(Ne,model=(dv,die),threshold=threshold,reset=reset,init=(reset,0*mV/ms))
Pi=NeuronGroup(Ni,model=(dv,die2),threshold=threshold,reset=reset,init=(reset,0*mV/ms))
Ce=Connection(Pe,Pe,'v')
Ce.connect_random(Pe,Pe,.1,lambda :Je*uniform())
Ce=Connection(Pe,Pi,'v')
Ce.connect_random(Pe,Pi,.1,lambda :Je*uniform())
Ci=Connection(Pi,Pe,'v')
Ci.connect_random(Pi,Pe,.1,lambda :Ji*uniform())
Ci=Connection(Pi,Pi,'v')
Ci.connect_random(Pi,Pi,.1,lambda :Ji*uniform())

duration=400*ms

#print 'mu(nu=5Hz)=%f' % (mb+Ne*Je*.015-leak,)
#print 'mu(nu=100Hz)=%f' % (mb+Ne*Je*.1-leak,)

V0=StateMonitor(Pe,'v',record=0)
S=SpikeMonitor(Pe,record=True)

t2 = time.time()
print 'Elapsed time is ', str(t2-t1), ' seconds.'

t1 = time.time()

run(150*ms)
mue = 6.5*mV/ms
sigmae = 7.5*mV/ms
run(300*ms)
mue = mb
sigmae = sb
run(50*ms)

t2 = time.time()
print 'Elapsed time is ', str(t2-t1), ' seconds.'
print S.nspikes

# -------------------------------------------------------------------------
# Plot everything
# -------------------------------------------------------------------------


def myplot():
    
    t1 = time.time()

    figure()
    
    # Membrane potential trace of the zeroeth neuron
    subplot(3,1,1)
    
    t=V0.times
    vt=V0[0]
    vt[vt>=20.0*mV]=65.0*mV
    plot(t,vt)
    #ylabel(r'$V-V_{rest}\ \left[\rm{mV}\right]$')
    
    # Raster plot of the spikes of the network
    subplot(3,1,2)
    myfirings = array(S.spikes)
    myfirings_100 = myfirings[myfirings[:,0]<min(100,Ne)]
    plot(myfirings_100[:,1],myfirings_100[:,0],'.')
    axis([0*ms, duration, 0, min(100,Ne)])
    ylabel('Neuron index')
    
    # Mean firing rate of the excitatory population as a function of time
    subplot(3,1,3)
    # 1 ms resultion of rate histogram
    dx = 1.0*ms
    x = arange(0*ms,duration,dx)
    myfirings_Ne = myfirings[myfirings[:,0]<Ne]
    mean_fe,x = histogram(myfirings_Ne[:,1],x)
    plot(x,mean_fe/dx/Ne*1000.0,ls='steps')
    ylabel('Hz')
    xlabel('time [ms]')
    t2 = time.time()
    print 'Finished.  Elapsed', str(t2-t1), ' seconds.'



myplot()
show()