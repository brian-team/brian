from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

hrtfdb = IRCAM_LISTEN(r'D:\HRTF\IRCAM')
subject = 1002
hrtfset = hrtfdb.load_subject(subject)
hrtf = hrtfset.hrtf[0]

x = Sound(randn(10000, 1), samplerate=44100*Hz)
y = hrtf.apply(x)

fb = FIRFilterbank(x, hrtf.impulseresponse.T)
fb.buffer_init()
print 'First fetch'
import time
start = time.time()
z1 = fb.buffer_fetch(0, 5000)
print 'Second fetch'
print 'time:', time.time()-start
z2 = fb.buffer_fetch(5000, 7500)
print 'Third fetch'
z3 = fb.buffer_fetch(7500, 10000)
print 'Finished fetching'
z = vstack((z1, z2, z3))

print z.shape

print amax(abs(z-y))

subplot(211)
plot(x)
subplot(212)
plot(y)
show()
