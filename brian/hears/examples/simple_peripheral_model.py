
'''
<<<<<<< .mine
Implemenation of a simple peripheral model consisting of a bank of gammatone filters
followed by  half wave rectification and compression
=======
Implementation of a simple peripheral model consisting of a bank of gammatone filters
followed by  half-wave rectification and compression
>>>>>>> .r2038
'''

from brian import *
from brian.hears import*
from brian.hears import filtering
filtering.use_gpu = False
samplerate=44*kHz
defaultclock.dt = 1/samplerate
simulation_duration=50*ms
sound = whitenoise(simulation_duration,samplerate).ramp()
nbr_center_frequencies=5
center_frequencies=erbspace(100*Hz,1000*Hz, nbr_center_frequencies) 


# bank of gammatone filters
gammatone_filterbank= GammatoneFilterbank(samplerate,center_frequencies)


#half way rectification + compression
v=1./3  #compression exponent
func=lambda x:(clip(x,0,Inf)**v)
rectification_and_compression=FunctionFilterbank(samplerate, nbr_center_frequencies, func)

#the whole path is the two previous filterbank in chain
peripheral_filter=FilterbankChain([gammatone_filterbank,rectification_and_compression])


peripheral= FilterbankGroup(peripheral_filter, sound)

peripheral_monitor = StateMonitor(peripheral, 'output', record=True)

run(simulation_duration)

time_axis=peripheral_monitor.times

figure()
suptitle('Outputs of the peripheral model (per frequency channel)')
for ifrequency in range((nbr_center_frequencies)):
    subplot(nbr_center_frequencies,1,ifrequency+1)
    plot(time_axis*1000,peripheral_monitor [ifrequency])
    xlabel('(ms)')
    
show()


