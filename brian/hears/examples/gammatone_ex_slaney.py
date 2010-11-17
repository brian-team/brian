
from brian import *
from brian.hears import*
from brian.hears import filtering
from time import time
filtering.use_gpu = False 
from scipy.io import savemat
set_global_preferences(useweave=False)
#set_global_preferences(weavecompiler ='gcc')


samplerate=44*kHz
defaultclock.dt = 1/samplerate

simulation_duration=50*ms

dBlevel=50  # dB level in rms dB SPL

samplerate,sound=get_wav('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
sound=Sound(sound,samplerate)
sound=sound.setintensity(dBlevel)
#sound.atintensity(dBlevel)
print 'fs=',samplerate,'duration=',len(sound)/samplerate
simulation_duration=len(sound)/samplerate
defaultclock.dt = 1/samplerate
#plot(sound)

#sound = whitenoise(simulation_duration,samplerate).ramp()

b1=1.81
nbr_center_frequencies=50
#center_frequencies=erbspace(100*Hz, 1000*Hz, nbr_center_frequencies)
center_frequencies=log_space(100*Hz, 1000*Hz, nbr_center_frequencies)

gammatone =GammatoneFilterbank(samplerate,center_frequencies,b=b1 )
gammatone_group = FilterbankGroup(gammatone, sound)

gt_mon = StateMonitor(gammatone_group, 'output', record=True)
t1=time()
run(simulation_duration)
print 'the simulation took',time()-t1,' seconds to run'
time_axis=gt_mon.times

brian_hears=gt_mon.getvalues()
data=dict()
data['out']=brian_hears
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/gT_slaney_BH.mat',data)

#figure()
#imshow(flipud(gt_mon.getvalues()),aspect='auto')
#show()




    