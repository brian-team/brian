"""

This appears to be an accurate model of the time spent in the threshold
calling mechanism, and so can be used as a basis for optimising.

With netsize = 4000 and repeats = 400000 as in the cuba-profile.txt example,
using original method gives 40s in nonzero method of numpy.ndarray, which
accords with what happens in the CUBA profile, and 11s in
FakeThreshold.__call__ which also accords.

"""

from numpy import *
import c_profile
import time

## parameters for comparison with cuba-profile.txt
#netsize = 4000
#repeats = 400000
# parameters for quicker runs to try out optimisations
# original method: 4.768s in nonzero, 1.366 in __call__
netsize=4000
repeats=50000


class FakeNeuronGroup(object):
    pass


class FakeThreshold(object):
    # this is the original method
    def __call__(self, P):
        return ((P.S[0]>self.Vt).nonzero())[0]

V=random.uniform(size=netsize)
Vt=1-0.0005

P=FakeNeuronGroup()
for i in range(40):
    exec 'P.a'+str(i)+'=1'
P.S=zeros((3, netsize))
P.S[0]=V

T=FakeThreshold()
#T.Vt = Vt
for i in range(40):
    exec 'T.a'+str(i)+'=1'

def main():
    for i in xrange(repeats):
        spikes=T(P)
c_profile.run('main()')

#def timeforfiringfraction(Vt):
#    global T
#    T.Vt = Vt
#    start = time.time()
#    for i in xrange(repeats):
#        spikes = T(P)
#    return time.time()-start
#
#def main():
#    for rate in [1,5,10,50,100,400,4000]:
#        Vt = 1 - float(rate)/4000.
#        print rate, ",", timeforfiringfraction(Vt)
#        
#c_profile.run('main()')
