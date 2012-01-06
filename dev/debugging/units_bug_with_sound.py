#import brian_no_units
from brian import *
from brian.hears import *

print second.__class__

snd = silence(1 * second)
snd[0*second:0.5*second] = 0
snd[0:100] = 0
snd[0*second:0.5*volt] = 0
