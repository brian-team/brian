from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

hrtfdb = IRCAM_LISTEN(r'D:\HRTF\IRCAM')
subject = 1002
hrtfset = hrtfdb.load_subject(subject)
hrtf = hrtfset.hrtf[0]

x = Sound(randn(22050, 1), samplerate=44100*Hz)

fb = hrtfset.filterbank(x,
                        interleaved=True
                        )
#fb.minimum_buffer_size = 500
print fb.minimum_buffer_size
fb.buffer_init()
import time
start = time.time()
z = fb.buffer_fetch(0, 22050)
#z1 = fb.buffer_fetch(0, 250)
#z2 = fb.buffer_fetch(250, 500)
#z3 = fb.buffer_fetch(500, 750)
#z4 = fb.buffer_fetch(750, 1000)
#z = vstack((z1, z2, z3, z4))
print 'time:', time.time()-start

L = array(hrtf.apply(x).left).flatten()
l = z[:, 0]

R = array(hrtf.apply(x).right).flatten()
r = z[:, 1] # interleaved
#r = z[:, 187] # serial

print amax(abs(L-l)), amax(abs(R-r))

plot(l)
plot(r)
show()
