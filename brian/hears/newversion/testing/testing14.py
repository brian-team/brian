from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

print dB, dB.__class__
x = 3*dB
print x, x.__class__
print -x
print abs(x)
print x+1
