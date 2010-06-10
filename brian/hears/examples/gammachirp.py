'''
Plot the response to a click of a gammatone filter and  a gammachirp filter
'''

from brian import *
from brian.hears import*
from brian.hears import filtering
filtering.use_gpu = False 


set_global_preferences(useweave=True)
set_global_preferences(weavecompiler ='gcc')

samplerate=44*kHz
defaultclock.dt = 1/samplerate

simulation_duration=25*ms

#sound = whitenoise(simulation_duration,samplerate).ramp()
sound = click(.1*ms, amplitude=20,rate=samplerate)

center_frequencies=array([1000*Hz])
c=-1  #c determines the rate of the frequency modulation or the chirp rate (c=0 is a gammtone filter)

gammachirp =GammachirpFilterbankIIR(samplerate,center_frequencies,c=c )
gammatone =GammatoneFilterbank(samplerate,center_frequencies )

gammachirp_group = FilterbankGroup(gammachirp, sound)
gammatone_group = FilterbankGroup(gammatone, sound)

gc_mon = StateMonitor(gammachirp_group, 'output', record=True)
gt_mon = StateMonitor(gammatone_group, 'output', record=True)


run(simulation_duration)

time_axis=gc_mon.times
number_samples=len(time_axis)
frequency_axis=linspace(0,1,number_samples-1)*samplerate
figure()
subplot(211)
plot(time_axis*1000,gc_mon[0], label='Gammachirp')
plot(time_axis*1000,gt_mon [0],label='Gammatone')
legend()
xlabel('(ms)')
title('Time response to an impulse')

subplot(212)
frequency_response_gc=20*log10(abs(fft(gc_mon[0])))
frequency_response_gt=20*log10(abs(fft(gt_mon[0])))
plot(frequency_axis[:number_samples/4],frequency_response_gc[:number_samples/4])
plot(frequency_axis[:number_samples/4],frequency_response_gt[:number_samples/4])
title('Frequency response to an impulse')
xlabel('(Hz)')
ylabel('dB')
show()
    