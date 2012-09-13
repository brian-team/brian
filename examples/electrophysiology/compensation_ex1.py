#!/usr/bin/env python
"""
Example of L^p electrode compensation method. Requires binary files 
"current.npy" and "rawtrace.npy".

    Rossant et al., "A calibration-free electrode compensation method"
    J. Neurophysiol 2012
"""
import os

from brian import *
import numpy as np
from brian.library.electrophysiology import *

working_dir = os.path.dirname(__file__)

# load data
dt = 0.1*ms
current = np.load(os.path.join(working_dir, "current.npy"))  # 10000-long vector, 1s duration
rawtrace = np.load(os.path.join(working_dir, "trace.npy"))  # 10000-long vector, 1s duration
t = linspace(0., 1., len(current))

# launch compensation
r = Lp_compensate(current, rawtrace, dt, p=1.0, full=True)

# print best parameters
print "Best parameters: R, tau, Vr, Re, taue:"
print r["params"]

# plot traces
subplot(211)
plot(t, current, 'k')

subplot(212)
plot(t, rawtrace, 'k')  # raw trace
plot(t, r["Vfull"], 'b')  # full model trace (neuron and electrode)
plot(t, r["Vcompensated"], 'g')  # compensated trace

show()