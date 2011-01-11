from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
from scipy.io import savemat
from time import time

dBlevel=50*dB  # dB level in rms dB SPL
#sound=Sound.load('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')

sound=whitenoise(100*ms,samplerate=44*kHz).ramp()
sound=sound.atlevel(dBlevel)

print 'fs=',sound.samplerate,'duration=',len(sound)/sound.samplerate

simulation_duration=len(sound)/sound.samplerate

nbr_center_frequencies=5

c=0.3 #linspace(-2,2,nbr_center_frequencies)#-2.96 
time_constant=linspace(0.6,0.1,nbr_center_frequencies)*ms

cf=log_space(3000*Hz, 5000*Hz, nbr_center_frequencies)

gamma_chirp=LinGammachirpFilterbank(sound, cf,time_constant,c)


gamma_chirp.buffer_init()
t1=time()
gamma_chirp_mon=gamma_chirp.buffer_fetch(0, len(sound))
print 'the simulation took',time()-t1,' seconds to run'


data=dict()
data['out']=gamma_chirp_mon.T
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/gT_gc_BH.mat',data)

figure()
for ich in xrange(nbr_center_frequencies):
    subplot(nbr_center_frequencies,1,ich+1)
    plot(gamma_chirp.impulse_response[ich,:])  

figure()
imshow(gamma_chirp_mon.T,aspect='auto')    
 
#imshow(flipud(gamma_chirp_mon.T),aspect='auto')    
show()    