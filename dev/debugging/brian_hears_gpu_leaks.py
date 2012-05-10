from brian import *
set_global_preferences(brianhears_usegpu=True)
from brian.hears import *

sound = silence(1 * second)
gfb = Gammatone(sound, 100)
for i in xrange(1000):
    gfb.process()
    print i
