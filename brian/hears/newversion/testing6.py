from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *

hrtfdb = IRCAM_LISTEN(r'D:\HRTF\IRCAM')
subject = 1002
hrtfset = hrtfdb.load_subject(subject)
hrtf = hrtfset.hrtf[0]

x = Sound(randn(1000, 1), samplerate=44100*Hz)
y = hrtf.apply(x)

subplot(211)
plot(x)
subplot(212)
plot(y)
show()
