from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *

simulation_duration=100*ms
samplerate=44*kHz

sound = whitenoise(simulation_duration,samplerate).ramp()

nbr_cf=50
cf=log_space(100*Hz, 1000*Hz, nbr_cf)

c1=-2.96
b1=1.81

pGc=LogGammachirpFilterbank(sound,cf,b=b1, c=c1)
 
cf_test_ind = arange(0,len(cf))

pGc_test=RestructureFilterbank(pGc,indexmapping=cf_test_ind )

pGc_signal=pGc.buffer_fetch(0, len(sound))

pGc_test=pGc_test.buffer_fetch(0, len(sound))

figure()
imshow(flipud(pGc_signal.T),aspect='auto')  

figure()
imshow(flipud(pGc_test.T),aspect='auto')  
show()




