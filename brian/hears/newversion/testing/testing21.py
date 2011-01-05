from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

x = whitenoise(100*ms, nchannels=13)
x /= amax(abs(x))

hrtfset = ITDDatabase(13).load_subject()

fb = hrtfset.filterbank(Repeat(x, 2), interleaved=True)

y = fb.fetch(x.duration)
y += reshape(repeat(arange(13), 2), (1, 26))

plot(y)
show()
