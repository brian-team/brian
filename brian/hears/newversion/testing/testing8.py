from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

hrtfdb = IRCAM_LISTEN(r'D:\HRTF\IRCAM')
subject = 1002
hrtfset = hrtfdb.load_subject(subject)
hrtf = hrtfset.hrtf[0]

x = Sound(randn(1000, 1), samplerate=44100*Hz)

fb = hrtfset.filterbank(x,
                        interleaved=True
                        )
print fb.nchannels
fb.buffer_init()
import time
start = time.time()
z = fb.buffer_fetch(0, 1000)
print 'time:', time.time()-start

L = array(hrtf.apply(x).left).flatten()
l = z[:, 0]

R = array(hrtf.apply(x).right).flatten()
r = z[:, 1] # interleaved
#r = z[:, 187] # serial

print L.shape, l.shape, z.shape
print amax(abs(L-l)), amax(abs(R-r))

plot(l)
plot(r)
show()
