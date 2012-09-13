#!/usr/bin/env python
'''
Example of using :class:`RemoteControlServer` and :class:`RemoteControlClient`
to control a simulation as it runs in Brian.

Run the script remotecontrolserver.py before running this.
'''
from brian import *
import time

client = RemoteControlClient()

time.sleep(1)

subplot(121)
plot(*client.evaluate('(M.times, M.values)'))

client.execute('G.I = 1.1')

time.sleep(1)

subplot(122)
plot(*client.evaluate('(M.times, M.values)'))

client.stop()

show()
