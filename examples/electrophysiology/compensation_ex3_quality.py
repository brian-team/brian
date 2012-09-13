#!/usr/bin/env python
"""
Example of quality check method. Requires binary files 
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
compensatedtrace = np.load(os.path.join(working_dir, "compensatedtrace.npy"))  # obtained with example1
t = linspace(0., 1., len(current))

# get trace quality of both raw and compensated traces
r = get_trace_quality(rawtrace, current, full=True)
rcomp = get_trace_quality(compensatedtrace, current, full=True)
spikes = r["spikes"]
print "Quality coefficient for raw: %.3f and for compensated trace: %.3f" % \
      (r["correlation"], rcomp["correlation"])

# plot trace and spikes
plot(t, rawtrace, 'k')
plot(t, compensatedtrace, 'g')
plot(t[spikes], rawtrace[spikes], 'ok')
plot(t[spikes], compensatedtrace[spikes], 'og')
show()
