"""
This example uses as input a sound (white noise) which is generated online and a time shifted version of it. This way very long simulations with small time step can be
performed without running out of memory.

shift is defined with a lambda function which will be processed every time_interval.  time_interval is negative per default so that there are never any h
any change in the shift which then can be defined as shift=lambda:3*ms for example to have a constant shift
"""


from brian import *
from brian.hears import*
from brian.hears import filtering
filtering.use_gpu = False 
set_global_preferences(useweave=True)
set_global_preferences(weavecompiler ='gcc')


samplerate=44*kHz
defaultclock.dt = 1/samplerate

simulation_duration=500*ms
max_abs_itd=10*ms
shift='variable' #1*ms
time_interval=10*ms
shift=lambda :10*randn(1)*ms

sound_left  = OnlineWhiteNoiseBuffered(samplerate,0,1,max_abs_itd)
sound_right  = OnlineWhiteNoiseShifted(samplerate,sound_left,shift=shift,time_interval=time_interval)



nbr_center_frequencies=5
center_frequencies=erbspace(200*Hz, 2*kHz, nbr_center_frequencies)

gammatone_left =GammatoneFilterbank(samplerate,center_frequencies )
gammatone_group_left  = FilterbankGroup(gammatone_left, sound_left)

gammatone_right =GammatoneFilterbank(samplerate,center_frequencies )
gammatone_group_right  = FilterbankGroup(gammatone_right, sound_right)

gt_mon_left = StateMonitor(gammatone_group_left, 'output', record=True)
gt_mon_right = StateMonitor(gammatone_group_right, 'output', record=True)

run(simulation_duration)

time_axis=gt_mon_left.times

figure()
suptitle('Outputs of the gammatone filterbank')
for ifrequency in range((nbr_center_frequencies)):
    subplot(nbr_center_frequencies,1,ifrequency+1)
    plot(time_axis*1000,gt_mon_left [ifrequency])
    plot(time_axis*1000,gt_mon_right[ifrequency])
    xlabel('(ms)')
    
show()