'''
CF-dependence of compressive nonlinearity in the Tan&Carney model.
Reproduces Fig. 7 from:

Tan, Q., and L. H. Carney.
    "A Phenomenological Model for the Responses of Auditory-nerve Fibers.
    II. Nonlinear Tuning with a Frequency Glide".
    The Journal of the Acoustical Society of America 114 (2003): 2007.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from brian import *
#set_global_preferences(useweave=True)
from brian.hears import *
from brian.hears.filtering.tan_carney import TanCarneySignal, MiddleEar

samplerate = 50*kHz
set_default_samplerate(samplerate)
duration = 50*ms

def product(*args):
    # Simple (and inefficient) variant of itertools.product that works for
    # Python 2.5 (directly returns a list instead of yielding one item at a
    # time)
    pools = map(tuple, args)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

def gen_tone(freq, level):
    ''' 
    Little helper function to generate a pure tone at frequency `freq` with
    the given `level`. The tone has a duration of 50ms and is ramped with
    two ramps of 2.5ms.
    '''
    freq = float(freq) * Hz
    level = float(level) * dB    
    return tone(freq, duration).ramp(when='both',
                                     duration=2.5*ms,
                                     inplace=False).atlevel(level)

freqs = [500, 1100, 2000, 4000]
levels = np.arange(-10, 100.1, 5)
cf_level = product(freqs, levels)

# steady-state
start = 10*ms*samplerate
end = 45*ms*samplerate

# For Figure 7 we have manually adjusts the gain for different CFs -- otherwise
# the RMS values wouldn't be identical for low CFs. Therefore, try to estimate
# suitable gain values first using the lowest CF as a reference
ref_tone = gen_tone(freqs[0], levels[0])
F_out_reference = TanCarneySignal(MiddleEar(ref_tone, gain=1), freqs[0],
                                  update_interval=1).process().flatten()

ref_rms = np.sqrt(np.mean((F_out_reference[start:end] -
                           np.mean(F_out_reference[start:end]))**2))

gains = np.linspace(0.1, 1, 50) # for higher CFs we need lower gains
cf_gains = product(freqs[1:], gains)
tones = Sound([gen_tone(freq, levels[0]) for freq, _ in cf_gains])
F_out_test = TanCarneySignal(MiddleEar(tones, gain=np.array([g for _, g in cf_gains])),
                             [cf for cf,_  in cf_gains], update_interval=1).process()

reshaped_Fout = F_out_test.T.reshape((len(freqs[1:]), len(gains), -1))
rms = np.sqrt(np.mean((reshaped_Fout[:, :, start:end].T -
                       np.mean(reshaped_Fout[:, :, start:end], axis=2).T).T**2,
                       axis=2))

# get the best gain for each CF using simple linear interpolation
gain_dict = {freqs[0]: 1.} # reference gain
for idx, freq in enumerate(freqs[1:]):
    gain_dict[freq] = interp1d(rms[idx, :], gains)(ref_rms)

# now do the real test: tones at different levels for different CFs
tones = Sound([gen_tone(freq, level) for freq, level in cf_level])
F_out = TanCarneySignal(MiddleEar(tones,
                                  gain=np.array([gain_dict[cf] for cf, _ in cf_level])),
                        [cf for cf, _ in cf_level],
                        update_interval=1).process()

reshaped_Fout = F_out.T.reshape((len(freqs), len(levels), -1))

rms = np.sqrt(np.mean((reshaped_Fout[:, :, start:end].T -
                      np.mean(reshaped_Fout[:, :, start:end], axis=2).T).T**2,
                      axis=2))

# This should more or less reproduce Fig. 7
plt.plot(levels, rms.T)
plt.legend(['%.0f Hz' % cf for cf in freqs], loc='best')
plt.xlim(-20, 100)
plt.ylim(1e-6, 1)
plt.yscale('log')
plt.xlabel('input signal SPL (dB)')
plt.ylabel('rms of AC component of Fout')
plt.show()
