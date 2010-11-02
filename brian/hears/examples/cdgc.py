
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
from time import  time


from brian.hears import*
from brian.hears import filtering
filtering.use_gpu = False
#set_global_preferences(useweave=True)

samplerate=100*kHz
defaultclock.dt = 1/samplerate
simulation_duration=50*ms
dBlevel=50  # dB level in rms dB SPL
sound = whitenoise(simulation_duration,samplerate,dB=dBlevel).ramp()
nbr_cf=500
center_frequencies=erbspace(100*Hz,1000*Hz, nbr_cf) 

#### middle ear processing ####
f_cut_off1=25*kHz/ ( ( 10**(0.1*10)-1)**(1.0/(2.0*2))) #Find the butterworth natural frequency W0 from the lower and higher cutt off 
f_cut_off2=30*kHz/ ( ( 10**(0.1*10)-1)**(1.0/(2.0*2)))
lp_l1=ButterworthFilterbank(samplerate, 1, 2,f_cut_off1, btype='low')
lp_l2=ButterworthFilterbank(samplerate, 1, 3,f_cut_off2, btype='low')
middle_ear=FilterbankChain([lp_l1,lp_l2])


#### conversion from pressure (Pascal) in stapes velocity (in m/s)
stape_scale=0.00014
stape_scale=1./0.00014
stapes_fun=lambda x:x*stape_scale
stapes_conv=FunctionFilterbank(samplerate, 1, stapes_fun)


#### Control Path ####

#bandpass filter (second order  gammatone filter)
cf_pgc_control=10**(0.339+0.895*log10(center_frequencies))
glide_slope_control
time_cst_control

bandpass_control= GammachirpFilterbankFIR(samplerate,cf_pgc_control,glide_slope_control,time_cst_control)

# level measurement site 1
func_level1=lambda x:g*x
level1=FunctionFilterbank(samplerate, nbr_center_frequencies, func_level1)

#low pass filter(cascade of 4 second order lowpass butterworth filters)
cutoff_frequencies_control=cf_pgc_control
order_lowpass_linear=2
lp_l=ButterworthFilterbank(samplerate, nbr_cf, order_lowpass_linear, cutoff_frequencies_linear, btype='low')
lowpass_linear=CascadeFilterbank(lp_l,4)

# level measurement site 2
func_level2=lambda x:g*x
level2=FunctionFilterbank(samplerate, nbr_center_frequencies, func_level2)

#signal pathway
signal_path=FilterbankChain([bandpass_linear,lowpass_linear])


#### Signal Path ####

#bandpass filter (third order gammatone filters)
cf_pgc_signal=10**(0.339+0.895*log10(center_frequencies))
glide_slope_signal
time_cst_signal

bandpass_signal= GammachirpFilterbankFIR(samplerate,cf_pgc_signal,glide_slope_signal,time_cst_signal)

#high pass filter
cutoff_frequencies_nonlinear=center_frequencies_nonlinear
order_highpass_nonlinear=2
def level_dep_hp_filter(level1,level2):
    pass

hp_signal=TimeVaryingIIRFilterbank2(sound.rate,nbr_cf)
lowpass_nonlinear=CascadeFilterbank(lp_nl,4)

#nonlinear pathway
nonlinear_path=FilterbankChain([bandpass_nonlinear1,compression,bandpass_nonlinear2,lowpass_nonlinear])


#### adding the two pathways together ####
dnrl_filter=linear_path+nonlinear_path

#### chaining everything ####
filters_1dpath=FilterbankChain([middle_ear,stapes_conv])
processing_1dpath= FilterbankGroup(filters_1dpath, sound)     #1d fu=ilter chain of middle ear

dnrl= FilterbankGroup(dnrl_filter, processing_1dpath.output)

#dnrl= FilterbankGroup(dnrl_filter, sound)
dnrl_monitor = StateMonitor(dnrl, 'output', record=True)

t1=time()
run(simulation_duration)
print 'the simulation took %f sec to run' %(time()-t1)

time_axis=dnrl_monitor.times

figure()
for ifrequency in range((nbr_cf)):
    subplot(nbr_cf,1,ifrequency+1)
    plot(time_axis*1000,dnrl_monitor [ifrequency])
    xlabel('(ms)')
    
show()


