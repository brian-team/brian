
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
cf=erbspace(100*Hz,1000*Hz, nbr_cf) 
order=4
c1=-2.96
b1=1.81
c2=2.2
b2=2.17

ERBrate= 21.4*log10(4.37*cf/1000+1)
ERBwidth= 24.7*(4.37*cf/1000 + 1)

fp1 = cf + c1*ERBwidth*b1/order


#### Control Path ####

#bandpass filter (second order  gammatone filter)
pGc= GammachirpFilterbankIIR(samplerate, fr, c=c1,b=b1)


#low pass filter(cascade of 4 second order lowpass butterworth filters)
cutoff_frequencies_control=cf_pgc_control
order_lowpass_linear=2
lp_l=ButterworthFilterbank(samplerate, nbr_cf, order_lowpass_linear, cutoff_frequencies_linear, btype='low')
lowpass_linear=CascadeFilterbank(lp_l,4)

#signal pathway
class FilterCoeffUpdate:
    def __init__(self, fs,nbr_channel,s1,s2,):
  
    def __call__(self):



signal_path=FilterbankChain([bandpass_linear,lowpass_linear])


#### Signal Path ####

#bandpass filter (third order gammatone filters)
cf_pgc_signal=10**(0.339+0.895*log10(cf))
glide_slope_signal
time_cst_signal

bandpass_signal= GammachirpFilterbankFIR(samplerate,cf_pgc_signal,glide_slope_signal,time_cst_signal)

#high pass filter
cutoff_frequencies_nonlinear=cf_nonlinear
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


