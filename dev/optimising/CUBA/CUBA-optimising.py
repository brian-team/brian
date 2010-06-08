"""
CUBA optimising
"""

import BrianNoUnits
from brian import *
from brian.stdunits import *
from brian.globalprefs import *
from scipy import rand
import pylab
import time
import c_profile
import profile

def main():
    taum = 20 * ms
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV
    Vr = -60 * mV
    dv = lambda v, ge, gi: (ge + gi - (v + 49 * mV)) / taum
    dge = lambda v, ge, gi:-ge / taue
    dgi = lambda v, ge, gi:-gi / taui
    P = NeuronGroup(16000, model=(dv, dge, dgi), threshold=Vt, \
                  reset=Vr, init=(0 * volt, 0 * volt, 0 * volt))
    Pe = P.subgroup(12800)
    Pi = P.subgroup(3200)
    Ce = Connection(Pe, P, 1)
    Ci = Connection(Pi, P, 2)
    we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight
    Ce.connect_random(Pe, P, 0.02, weight=we)
    Ci.connect_random(Pi, P, 0.02, weight=wi)
    P['v'] = Vr + rand(1, len(P)) * (Vt - Vr)    # Or P.S[0]=...

    net = MagicNetwork()
    start = time.time()
    net.run(10 * second)
    print time.time() - start


set_global_preferences(useweave=True)

main()
#c_profile.run('main()')
#main()
