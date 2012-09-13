#!/usr/bin/env python
'''
Example of using :class:`RemoteControlServer` and :class:`RemoteControlClient`
to control a simulation as it runs in Brian.

After running this script, run remotecontrolclient.py or paste the code from
that script into an IPython shell for interactive control.
'''
from brian import *

eqs = '''
dV/dt = (I-V)/(10*ms)+0.1*xi*(2/(10*ms))**.5 : 1
I : 1
'''

G = NeuronGroup(3, eqs, reset=0, threshold=1)
M = RecentStateMonitor(G, 'V', duration=50*ms)

server = RemoteControlServer()

run(1e10*second)
