
from brian import *
from brian.hears import*
from brian.hears import filtering
filtering.use_gpu = False 
set_global_preferences(useweave=True)
set_global_preferences(weavecompiler ='gcc')


samplerate=44*kHz
defaultclock.dt = 1/samplerate

simulation_duration=50*ms
max_abs_itd=2*ms
shift=1*ms

sound_left  = OnlineWhiteNoiseBuffered(samplerate,0,1,max_abs_itd)
sound_right  = OnlineWhiteNoiseShifted(samplerate,sound_left,shift)

nbr_center_frequencies=5
center_frequencies=erbspace(200*Hz, 2*kHz, nbr_center_frequencies)

gammatone_left =GammatoneFilterbank(samplerate,center_frequencies )
gammatone_group_left  = FilterbankGroup(gammatone, sound_left)

gammatone_right =GammatoneFilterbank(samplerate,center_frequencies )
gammatone_group_right  = FilterbankGroup(gammatone, sound_right)

gt_mon_left = StateMonitor(gammatone_group_left, 'output', record=True)

run(simulation_duration)

time_axis=gt_mon.times

figure()
suptitle('Outputs of the gammatone filterbank')
for ifrequency in range((nbr_center_frequencies)):
    subplot(nbr_center_frequencies,1,ifrequency+1)
    plot(time_axis*1000,gt_mon_left [ifrequency])
    plot(time_axis*1000,gt_mon_right[ifrequency])
    xlabel('(ms)')
    
show()