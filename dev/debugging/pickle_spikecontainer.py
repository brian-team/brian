from brian import *
from brian.utils.circular import SpikeContainer
import pickle

def unpickle_SpikeContainer(m, allspikes):
    newsc = SpikeContainer(m)
    for spikes in allspikes[::-1]:
        newsc.push(spikes)
    return newsc

#class SpikeContainer(SpikeContainer):
#    def __init__(self, m, useweave=False, compiler=None):
#        super(SpikeContainer, self).__init__(m)
#        self.m = m
#    def __reduce__(self):
#        return (unpickle_SpikeContainer, (self.m, tuple(self[i].copy() for i in xrange(self.m))))

sc = SpikeContainer(10)

# push some spikes into sc

for i in xrange(100):
    sc.push(randint(1000, size=randint(5)))
    
print sc.lastspikes()
print sc[0], sc[1], sc[2], sc[9]

s = pickle.dumps(sc)

sc = pickle.loads(s)

print sc.lastspikes()
print sc[0], sc[1], sc[2], sc[9]
