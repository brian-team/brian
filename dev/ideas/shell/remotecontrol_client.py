from brian import *
from brian.experimental.remotecontrol import *
import time

client = RemoteControlClient()

def compute_rate(spikes):
    i, t = zip(*spikes)
    i = array(i)
    t = array(t)
    tmin = amin(t)
    tmax = amax(t)
    return len(i) / (tmax - tmin)

spikes = client.evaluate('M.spikes')
print 'Rate:', compute_rate(spikes)
client.execute('Ce.W.alldata[:]*=2')
time.sleep(5)
spikes = client.evaluate('M.spikes')
print 'Rate:', compute_rate(spikes)
client.execute('stop()')
