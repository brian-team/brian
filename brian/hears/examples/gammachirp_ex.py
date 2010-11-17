
from brian import *
from brian.hears import*
from brian.hears import filtering
from time import time
from scipy.io import savemat

filtering.use_gpu = False 
set_global_preferences(useweave=True)
set_global_preferences(weavecompiler ='gcc')


samplerate=44*kHz
defaultclock.dt = 1/samplerate

simulation_duration=50*ms

dBlevel=50  # dB level in rms dB SPL

samplerate,sound=get_wav('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
sound=Sound(sound,samplerate)
#sound.atintensity(dBlevel)
print 'fs=',samplerate,'duration=',len(sound)/samplerate
simulation_duration=len(sound)/samplerate
defaultclock.dt = 1/samplerate
#plot(sound)
c1=-2.96
b1=1.81
#sound = whitenoise(simulation_duration,samplerate).ramp()

nbr_center_frequencies=50
cf=erbspace(100*Hz, 1000*Hz, nbr_center_frequencies)
cf=log_space(100*Hz, 1000*Hz, nbr_center_frequencies)

gammatone =GammatoneFilterbank(samplerate,cf,b=b1 )
asym_comp=Asym_Comp_Filterbank(samplerate, cf, c=c1,asym_comp_order=4,b=b1)
pGc=FilterbankChain([gammatone,asym_comp])

#asym_comp2=Asym_Comp_Filterbank(samplerate, cf, c=2.17,asym_comp_order=4,b=2.2)
#pGc=FilterbankChain([gammatone,asym_comp,asym_comp2]) 
#pGc= GammachirpFilterbankIIR(samplerate, cf, c=c1,b=b1)
gammachirp_group = FilterbankGroup(pGc, sound)



gc_mon = StateMonitor(gammachirp_group, 'output', record=True)

t1=time()
run(simulation_duration)
print 'the simulation took',time()-t1,' seconds to run'
time_axis=gc_mon.times

brian_hears=gc_mon.getvalues()
#np.savetxt('/home/bertrand/Data/MatlabProg/AuditoryFilters/gT_meddis_BH.txt', brian_hears)
data=dict()
data['out']=brian_hears
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/gT_gc_BH.mat',data)
figure()
imshow(flipud(gc_mon.getvalues()),aspect='auto')
#suptitle('Outputs of the gammatone filterbank')
#for ifrequency in range((nbr_center_frequencies)):
#    subplot(nbr_center_frequencies,1,ifrequency+1)
#    plot(time_axis*1000,gt_mon [ifrequency])
#    xlabel('(ms)')
    
show()




    