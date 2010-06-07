from brian import *


class InputGenerator(object):
  """ A generator class to be passed to MultipleSpikeGeneratorGroup
  """
  def __init__(self, phase=0*ms, period=1*ms):
    self.phase=phase
    self.period=period
  def __call__(self):
    t=self.phase
    while(True):
      yield t
      t+=self.period

ig1=InputGenerator(3*ms, 10*ms)
ig2=InputGenerator(5*ms, 10*ms)
G=MultipleSpikeGeneratorGroup([ig1, ig2])

print G._threshold.spiketimes
print G._threshold.spiketimeiter

M=SpikeMonitor(G)
net=Network(G, M)
net.run(20*ms)
print M.spikes

net.reinit() # after reinit I don't see any firings

print defaultclock.t
print G._threshold.spiketimes
print G._threshold.spiketimeiter

ig1.phase=5*ms
ig2.phase=5*ms
net.run(20*ms)
print M.spikes
