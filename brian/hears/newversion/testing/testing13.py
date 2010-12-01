from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

x = whitenoise(1*second)
print x.level
x.level = 10*dB
x.level += 5*dB
print x.level

x = Sound((whitenoise(1*second), whitenoise(1*second)))

print x.level

x.level = (30, 5)

x = x.atlevel((20, 18))

x = x*gain(5*dB)

print x.level

plot(x)
show()
