from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

#x = whitenoise(1*second)
x = sequence((whitenoise(250*ms), 2.0*whitenoise(750*ms)))

# This class implements the gain (see Filterbank for details)
class GainFilterbank(Filterbank):
    def __init__(self, source, gain=1.0):
        Filterbank.__init__(self, source)
        self.gain = gain
    def buffer_apply(self, input):
        return self.gain*input

# This is the class for the updater object
class GainController(object):
    def __init__(self, target, target_rms, time_constant):
        self.target_rms = target_rms
        self.time_constant = time_constant
        self.target = target
    def reinit(self):
        self.sumsquare = 0
        self.numsamples = 0
    def __call__(self, input):
        T = input.shape[0]/self.target.samplerate
        self.sumsquare += sum(input**2)
        self.numsamples += input.size
        rms = sqrt(self.sumsquare/self.numsamples)
        g = self.target.gain
        g_tgt = self.target_rms/rms
        tau = self.time_constant
        self.target.gain = g_tgt+exp(-T/tau)*(g-g_tgt)

fb = GainFilterbank(x)

updater = GainController(fb, 0.2, 50*ms)

control = ControlFilterbank(fb, x, fb, updater, 10*ms)

#x[:] *= 10.0

y = control.buffer_fetch(0, len(x))

subplot(211)
plot(x)
plot(y)
subplot(212)
plot(y/x)
show()
