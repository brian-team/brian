from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

hrtfdb = IRCAM_LISTEN(r'D:\HRTF\IRCAM')
subject = 1002
hrtfset = hrtfdb.load_subject(subject)
hrtf = hrtfset.hrtf[0]

x = Sound(randn(1000, 1), samplerate=44100*Hz)
y = hrtf.apply(x)

fb = FIRFilterbank(x, hrtf.impulseresponse.T)
fb.buffer_init()
z = fb.buffer_fetch(0, 1000)

print z.shape

print amax(abs(z-y))

subplot(211)
plot(x)
subplot(212)
plot(y)
show()
