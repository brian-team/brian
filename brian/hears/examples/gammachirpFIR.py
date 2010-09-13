'''
linear gammachirp filter as a FIR implementation. The example is a model
of a NM neuron of a barn owl
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

nbr_channels=1
center_frequencies=4*kHz*ones(nbr_channels)
time_cst=0.2*ms#linspace(0.2,0.55,nbr_channels)*ms
glide_slope=-0.2#linspace(-0.2,0.2,nbr_channels)



gammachirp=GammachirpFilterbankFIR(samplerate,center_frequencies,glide_slope,time_cst)
    
gammachirp_group = FilterbankGroup(gammachirp, sound)
gc_mon = StateMonitor(gammachirp_group, 'output', record=True)


run(simulation_duration)

time_axis=gc_mon.times
number_samples=len(time_axis)
frequency_axis=linspace(0,1,number_samples-1)*samplerate
figure()
plot(time_axis*1000,gc_mon[0], label='Gammachirp')
legend()
xlabel('(ms)')
title('Time response to an impulse')

show()