from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

s1 = whitenoise(100*ms)
s2 = tone(1*kHz, 100*ms)

play([s1, s2], s1, s2, normalise=True, sleep=True)
