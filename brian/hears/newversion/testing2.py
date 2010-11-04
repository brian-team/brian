from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *

x = Sound(randn(1000, 2), rate=44100*Hz)

gfb = GammatoneFilterbank(x, [1*kHz, 2*kHz])

y = gfb.buffer_fetch(0, 1500)

for i in xrange(3):
    y = gfb.buffer_fetch(i*500, i*500+1000)
    
    print (y.shape,
           (gfb.cached_buffer_start,
            gfb.cached_buffer_end,
            gfb.cached_buffer_output.shape))

for i in xrange(x.nchannels):
    plot(y[:, i])
show()
