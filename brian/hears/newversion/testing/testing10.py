from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

#x = Sound((whitenoise(100*ms), whitenoise(100*ms)))

#x[:50*ms] = 0.1*Sound((whitenoise(50*ms), whitenoise(50*ms)))
#x[:50*ms, 0] = 0.1*whitenoise(50*ms)

x = whitenoise(100*ms)*Sound(linspace(0, 1, int(100*ms*(44.1*kHz))))#[:100*ms]
x = x[::-1]
#x[:50*ms] = 0.1*whitenoise(50*ms)

plot(x[:,0])
show()
