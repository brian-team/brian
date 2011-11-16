from brian import *
#This code leaks memory if brianhears_usegpu is switched on
set_global_preferences(brianhears_usegpu=True)
from brian.hears import *

sound = silence(1 * second)
gfb = Gammatone(sound, 100)
n = 1000
for i in xrange(n):
    if (i + 1) % 10 == 0:
        print '%d/%d' % (i+1, n)
    gfb.process()
    