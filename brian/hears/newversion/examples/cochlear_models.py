from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *

dBlevel=60*dB  # dB level in rms dB SPL
sound=Sound.load('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
samplerate=sound.samplerate
sound=sound.atlevel(dBlevel)
simulation_duration=len(sound)/samplerate
nbr_cf=50
cf=log_space(100*Hz, 1000*Hz, nbr_cf)

### DNRL
#param_drnl={}
#param_drnl['lp_nl_cutoff_m']=1.1
#dnrl_filter=DRNL(sound,cf,given_param=param_drnl)
#dnrl=dnrl_filter.buffer_fetch(0, len(sound))
#
### CDGC
#param_cdgc={}
#param_cdgc['c1']=-2.96
#interval=20
#cdgc_filter=CDGC(sound,cf,interval,given_param=param_cdgc)
#cdgc=cdgc_filter.buffer_fetch(0, len(sound))

## PMFR
param_pmfr={}
param_pmfr['fp1']=1.0854*cf-106.0034
interval=20
pmfr_filter=PMFR(sound,cf,interval,given_param=param_pmfr)
pmfr=pmfr_filter.buffer_fetch(0, len(sound))


figure()
#subplot(311)
#imshow(flipud(dnrl.T),aspect='auto')
#subplot(312)
#imshow(flipud(cdgc.T),aspect='auto')
subplot(313)
imshow(flipud(pmfr.T),aspect='auto')

show()