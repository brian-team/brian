#!/usr/bin/env python
"""
Coincidence detection example
=============================
Fig. 4 from:

    Rossant C, Leijon S, Magnusson AK, Brette R (2011).
    "Sensitivity of noisy neurons to coincident inputs".
    Journal of Neuroscience, 31(47).

Two distant or coincident spikes are injected into a noisy balanced
leaky integrate-and-fire neuron. The PSTH of the neuron in response to 
these inputs is calculated along with the extra number of spikes 
in the two cases. This number is higher for the coincident spikes,
showing the sensitivity of a noisy neuron to coincident inputs.
"""
from brian import *
import matplotlib.patches as patches
import matplotlib.path as path

def histo(bins, cc, ax):
    # get the corners of the rectangles for the histogram
    left = array(bins[:-1])
    right = array(bins[1:])
    bottom = zeros(len(left))
    top = bottom + cc
    
    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = array([[left,left,right,right], [bottom,top,top,bottom]]).T
    
    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)
    
    # make a patch out of it
    patch = patches.PathPatch(barpath, facecolor='blue', edgecolor='gray', alpha=0.8)
    ax.add_patch(patch)
    
    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

# neuron parameters
theta = -55*mV
vmean = -65*mV
taum = 5*ms
taue = 3*ms
taun = 15*ms
sigma = 4*mV

# input times
t1 = 100*ms
t2 = 120*ms

# simulation duration
dur = 200*ms

# number of neuron
N = 10000
bin = 2*ms

# EPSP size
int_EPSP=taue
int_EPSP2=taue*taue/(2*(taum+taue))
max_EPSP=(taum/taue)**(taum/(taue-taum))
we = 3.0*mV/max_EPSP

# model equations
eqs = '''
V=V0+noise : volt
dV0/dt=(-V0+psp)/taum : volt
dpsp/dt=-psp/taue : volt
dnoise/dt=(vmean-noise)/taun+sigma*(2./taun)**.5*xi : volt
'''
threshold = 'V>theta'
reset = vmean

# initialization of the NeuronGroup
reinit_default_clock()
group = NeuronGroup(2*N, model=eqs, reset=reset, threshold=threshold)
group.V0 = group.psp = 0*volt
group.noise = vmean + sigma * randn(2*N)

# input spikes
input_spikes = [(0, t1), (0, t2), (1, t1)]
input = SpikeGeneratorGroup(2, array(input_spikes))

# connections
C = Connection(input, group, 'psp')
C.connect_full(input[0], group[:N], weight=we)
C.connect_full(input[1], group[N:], weight=2*we)

# monitors
prM1 = PopulationRateMonitor(group[:N], bin=bin)
prM2 = PopulationRateMonitor(group[N:], bin=bin)

# launch simulation
run(dur)

# PSTH plot
figure(figsize=(10,10))
prMs = [prM1, prM2]
for i in [0,1]:
    prM = prMs[i]
    r = prM.rate[:-1]*bin
    m = mean(r[:len(r)/2])

    ax = subplot(211+i)
    histo(prM.times, r, ax)
    plot([0,dur],[m,m],'--r')
    title("%.2f extra spikes" % sum(r[t1/bin:(t2+20*ms)/bin]-m))
    xlim(.05, .2)
    ylim(0, .125)

show()
