'''
Example of the use of the cochlear models (:class:`~brian.hears.DRNL` and
:class:`~brian.hears.DCGC`) available in the library.
'''
from brian import *
from brian.hears import *

simulation_duration = 50*ms
set_default_samplerate(50*kHz)
sound = whitenoise(simulation_duration)
sound = sound.atlevel(50*dB) # level in rms dB SPL
cf = erbspace(100*Hz, 1000*Hz, 50) # centre frequencies

## DNRL
param_drnl = {}
param_drnl['lp_nl_cutoff_m'] = 1.1
drnl_filter=DRNL(sound, cf, type='human', param=param_drnl)
drnl = drnl_filter.process()

## DCGC
param_dcgc = {}
param_dcgc['c1'] = -2.96
interval = 1
dcgc_filter = DCGC(sound, cf, interval, param=param_dcgc)
dcgc = dcgc_filter.process()

figure()
subplot(211)
imshow(flipud(drnl.T), aspect='auto')
subplot(212)
imshow(flipud(dcgc.T), aspect='auto')
show()
