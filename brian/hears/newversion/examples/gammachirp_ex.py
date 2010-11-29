from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
from scipy.io import savemat
from time import time

dBlevel=50  # dB level in rms dB SPL
sound=Sound.load('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
samplerate=sound.samplerate
sound=sound.atintensity(dBlevel)
sound.samplerate=samplerate

print 'fs=',sound.samplerate,'duration=',len(sound)/sound.samplerate

simulation_duration=len(sound)/sound.samplerate


c1=-2.96
b1=1.81

nbr_center_frequencies=50
cf=erbspace(100*Hz, 1000*Hz, nbr_center_frequencies)
cf=log_space(100*Hz, 1000*Hz, nbr_center_frequencies)

asym_comp=GammachirpIIRFilterbank(sound,cf, c=c1,asym_comp_order=4,b=b1)

#gammatone =GammatoneFilterbank(sound,cf,b=b1 )
#asym_comp=Asym_Comp_Filterbank(gammatone, cf, c=c1,asym_comp_order=4,b=b1)

asym_comp.buffer_init()
t1=time()
asym_comp_mon=asym_comp.buffer_fetch(0, len(sound))
print 'the simulation took',time()-t1,' seconds to run'


data=dict()
data['out']=asym_comp_mon.T
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/gT_gc_BH.mat',data)

figure()
imshow(flipud(asym_comp_mon.T),aspect='auto')    
show()    