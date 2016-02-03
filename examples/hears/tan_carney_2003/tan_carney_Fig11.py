'''
Response area and phase response of a model fiber with CF=2200Hz in the 
Tan&Carney model. Reproduces Fig. 11 from:

Tan, Q., and L. H. Carney.
    "A Phenomenological Model for the Responses of Auditory-nerve Fibers.
    II. Nonlinear Tuning with a Frequency Glide".
    The Journal of the Acoustical Society of America 114 (2003): 2007.
'''

import matplotlib.pyplot as plt
import numpy as np

from brian import *
# set_global_preferences(useweave=True)
from brian.hears import *

def product(*args):
    # Simple (and inefficient) variant of itertools.product that works for
    # Python 2.5 (directly returns a list instead of yielding one item at a
    # time)
    pools = map(tuple, args)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

duration = 50*ms
samplerate = 50*kHz
set_default_samplerate(samplerate)
CF = 2200
freqs = np.arange(250.0, 3501., 50.)
levels = [10, 30, 50, 70, 90]
cf_level = product(freqs, levels)
tones = Sound([Sound.sequence([tone(freq * Hz, duration).atlevel(level*dB).ramp(when='both',
                                                                                duration=2.5*ms,
                                                                                inplace=False)])
               for freq, level in cf_level])

ihc = TanCarney(MiddleEar(tones), [CF] * len(cf_level), update_interval=2)
syn = ZhangSynapse(ihc, CF)
s_mon = StateMonitor(syn, 's', record=True, clock=syn.clock)
net = Network(syn, s_mon)
net.run(duration)

reshaped = s_mon.values.reshape((len(freqs), len(levels), -1))

# calculate the phase with respect to the stimulus
pi = np.pi
min_freq, max_freq = 1100, 2900
freq_subset = freqs[(freqs>=min_freq) & (freqs<=max_freq)]
reshaped_subset = reshaped[(freqs>=min_freq) & (freqs<=max_freq), :, :]
phases = np.zeros((reshaped_subset.shape[0], len(levels)))
for f_idx, freq in enumerate(freq_subset):
    period = 1.0 / freq
    for l_idx in xrange(len(levels)):
        phase_angles = np.arange(reshaped_subset.shape[2])/samplerate % period / period * 2*pi
        temp_phases = (np.exp(1j * phase_angles) *
                       reshaped_subset[f_idx, l_idx, :])
        phases[f_idx, l_idx] = np.angle(np.sum(temp_phases))

plt.subplot(2, 1, 1)
rate = reshaped.mean(axis=2)
plt.plot(freqs, rate)
plt.ylabel('Spikes/sec')
plt.legend(['%.0f dB' % level for level in levels], loc='best')
plt.xlim(0, 4000)
plt.ylim(0, 250)

plt.subplot(2, 1, 2)
relative_phases = (phases.T - phases[:, -1]).T
relative_phases[relative_phases > pi] = relative_phases[relative_phases > pi] - 2*pi
relative_phases[relative_phases < -pi] = relative_phases[relative_phases < -pi] + 2*pi 
plt.plot(freq_subset, relative_phases / pi)
plt.ylabel("Phase Re:90dB (pi radians)")
plt.xlabel('Frequency (Hz)')
plt.legend(['%.0f dB' % level for level in levels], loc='best')
plt.xlim(0, 4000)
plt.ylim(-0.5, 0.75)
plt.show()
