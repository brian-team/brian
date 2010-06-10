
'''
Implementation of the dual resonance nonlinear (DRNL) filter.
The parameters are those fitted for guinea-pigs
from Sumner et al., A nonlinear filter-bank model of the guinea-pig cochlear nerve:
Rate responses, JASA 2003

The entire pathway consists of the sum of a linear and a nonlinear pathway

The linear path consists of a bandpass function (second order gammatone), a low pass function,
and a gain/attenuation factor, g, in a cascade

The nonlinear path is  a cascade consisting of a bandpass function, a
compression function, a second bandpass function, and a low
pass function, in that order.

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

#Linear Pathway #########

#linear gain
g=10**(5.68-0.97*log10(center_frequencies))
func_gain=lambda x:g*x
gain=FunctionFilterbank(samplerate, nbr_center_frequencies, func_gain)

#bandpass filter (second order  gammatone filter)
center_frequencies_linear=10**(0.339+0.895*log10(center_frequencies))
bandwidth_linear=10**(1.3+0.53*log10(center_frequencies))
order_linear=2
bandpass_linear =MeddisGammatoneFilterbank(samplerate, center_frequencies_linear, order_linear, bandwidth_linear)

#low pass filter(cascade of 4 second order lowpass butterworth filters)
cutoff_frequencies_linear=center_frequencies_linear
order_lowpass_linear=2
lp_l=ButterworthFilterbank(samplerate, nbr_center_frequencies, order_lowpass_linear, cutoff_frequencies_linear, btype='low')
lowpass_linear=CascadeFilterbank(lp_l,4)

#linear pathway
linear_path=FilterbankChain([gain,bandpass_linear,lowpass_linear])



#Nonlinear Pathway#########

#bandpass filter (third order gammatone filters)
center_frequencies_nonlinear=center_frequencies
bandwidth_nonlinear=10**(0.8+0.58*log10(center_frequencies))
order_nonlinear=3
bandpass_nonlinear1=MeddisGammatoneFilterbank(samplerate, center_frequencies_nonlinear, order_nonlinear, bandwidth_nonlinear)

#compression (linear at low level, compress at high level)
a=10**(1.87+0.45*log10(center_frequencies))  #linear gain
b=10**(-5.65+0.875*log10(center_frequencies))  
v=0.1  #compression exponent
func_compression=lambda x:sign(x)*minimum(a*abs(x),b*abs(x)**v)
compression=FunctionFilterbank(samplerate, nbr_center_frequencies, func_compression)

#bandpass filter (third order gammatone filters)
bandpass_nonlinear2=MeddisGammatoneFilterbank(samplerate, center_frequencies_nonlinear, order_nonlinear, bandwidth_nonlinear)

#low pass filter
cutoff_frequencies_nonlinear=center_frequencies_nonlinear
order_lowpass_nonlinear=2
lp_nl=ButterworthFilterbank(samplerate, nbr_center_frequencies, order_lowpass_nonlinear, cutoff_frequencies_nonlinear, btype='low')
lowpass_nonlinear=CascadeFilterbank(lp_nl,4)

#nonlinear pathway
nonlinear_path=FilterbankChain([bandpass_nonlinear1,compression,bandpass_nonlinear2,lowpass_nonlinear])

##########
#adding the two pathways together
dnrl_filter=linear_path+nonlinear_path
dnrl= FilterbankGroup(dnrl_filter, sound)

dnrl_monitor = StateMonitor(dnrl, 'output', record=True)

run(simulation_duration)

time_axis=dnrl_monitor.times

figure()
for ifrequency in range((nbr_center_frequencies)):
    subplot(nbr_center_frequencies,1,ifrequency+1)
    plot(time_axis*1000,dnrl_monitor [ifrequency])
    xlabel('(ms)')
    
show()


