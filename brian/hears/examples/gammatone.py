
from brian import *
from brian.hears import*
from brian.hears import filtering
filtering.use_gpu = False 
set_global_preferences(useweave=True)
set_global_preferences(weavecompiler ='gcc')


samplerate=44*kHz
defaultclock.dt = 1/samplerate

simulation_duration=50*ms

sound = whitenoise(simulation_duration,samplerate).ramp()

nbr_center_frequencies=5
center_frequencies=erbspace(200*Hz, 2*kHz, nbr_center_frequencies)

gammatone =GammatoneFilterbank(samplerate,center_frequencies )
gammatone_group = FilterbankGroup(gammatone, sound)

gt_mon = StateMonitor(gammatone_group, 'output', record=True)

run(simulation_duration)

time_axis=gt_mon.times

figure()
suptitle('Outputs of the gammatone filterbank')
for ifrequency in range((nbr_center_frequencies)):
    subplot(nbr_center_frequencies,1,ifrequency+1)
    plot(time_axis*1000,gt_mon [ifrequency])
    xlabel('(ms)')
    
show()




    