from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
#from brian.hears.filtering import GammatoneFilterbank as oldGFB

x = Sound(randn(1000, 2), rate=44100*Hz)

gfb = GammatoneFilterbank(x, [1*kHz, 2*kHz])
#old_gfb = oldGFB(44100*Hz, [1*kHz, 2*kHz])
#y2 = old_gfb.apply(asarray(x))

ff = FunctionFilterbank(gfb, lambda input:clip(input, 0, Inf))

sfb = SumFilterbank((gfb, ff), (1, -1))

sfb = sfb*sfb

sfb.buffer_init()
y = sfb.buffer_fetch(0, 1500)

#for i in xrange(3):
#    y = gfb.buffer_fetch(i*500, i*500+1000)
#    
#    print (y.shape,
#           (gfb.cached_buffer_start,
#            gfb.cached_buffer_end,
#            gfb.cached_buffer_output.shape))

for i in xrange(x.nchannels):
    plot(y[:, i])
#    plot(y2[:, i])
show()
