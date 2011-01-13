'''
Example of the use of the cochlear models available in the library
'''
from brian import *
from brian.hears import *

simulation_duration=50*ms
samplerate=50*kHz
sound = whitenoise(simulation_duration,samplerate)
sound=sound.atlevel(50*dB) # dB level in rms dB SPL
cf=erbspace(100*Hz, 1000*Hz, 50)

## DNRL
param_drnl={}
param_drnl['lp_nl_cutoff_m']=1.1

drnl_filter=DRNL(sound,cf,type='human',param=param_drnl)
drnl=drnl_filter.process()

## CDGC
param_cdgc={}
param_cdgc['c1']=-2.96
interval=20
cdgc_filter=DCGC(sound,cf,interval,param=param_cdgc)
cdgc=cdgc_filter.process()

### PMFR
param_pmfr={}
param_pmfr['fp1']=1.0854*cf-106.0034
interval=20
pmfr_filter=PMFR(sound,cf,interval,param=param_pmfr)
pmfr=pmfr_filter.buffer_fetch(0, len(sound))

figure()
subplot(311)
imshow(flipud(drnl.T),aspect='auto')
subplot(312)
imshow(flipud(cdgc.T),aspect='auto')
subplot(313)
imshow(flipud(pmfr.T),aspect='auto')

show()