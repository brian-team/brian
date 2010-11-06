from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *

x = Sound(randn(1000, 10), rate=44100*Hz)

class AccumulateFilterbank(Filterbank):
    def __init__(self, source):
        Filterbank.__init__(self, source)
        self.nchannels = 1
    def buffer_apply(self, input):
        return reshape(sum(input, axis=1), (input.shape[0], 1))

fb = AccumulateFilterbank(x)

fb.buffer_init()

y = fb.buffer_fetch(0, 1000)

subplot(211)
plot(x)
subplot(212)
plot(y)
plot(sum(x, axis=1))
show()
