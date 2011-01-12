from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *

x = Sound(array([randn(1000), zeros(1000)]).T, samplerate=44100*Hz)
w = Sound(array([linspace(-1, 1, 1000), sin(linspace(0, 2*pi, 1000))]).T, samplerate=44100*Hz)
#w = Sound(sin(linspace(0, 2*pi, 1000)), samplerate=44100*Hz)

fb = RestructureFilterbank((x, w), 2, 'serial', 3)
#fb = RestructureFilterbank((x, w), 2, 'interleave', 3)
#fb = RestructureFilterbank((x, w), indexmapping=[0, 2, 3, 1])

fb.buffer_init()
y = fb.buffer_fetch(0, 1000)

subplot(211)
plot(x)
plot(w)
subplot(212)
for i, z in enumerate(y.T):
    plot(z+5*i)
show()
