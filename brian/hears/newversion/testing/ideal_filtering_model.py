# TODO: add gains to this model, to make it equivalent to original model
from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *
from plot_count import ircam_plot_count
import time

hrtfdb = IRCAM_LISTEN(r'D:\HRTF\IRCAM')
subject = 1002
hrtfset = hrtfdb.load_subject(subject)
index = randint(hrtfset.num_indices)
cfmin, cfmax, cfN = 150*Hz, 5*kHz, 40
sound = Sound.whitenoise(500*ms)

hrtf = hrtfset.hrtf[index]
cf = erbspace(cfmin, cfmax, cfN)
num_indices = hrtfset.num_indices

hrtf_fb = hrtf.filterbank(sound)
hrtfset_fb = hrtfset.filterbank(
        RestructureFilterbank(hrtf_fb, 
                indexmapping=repeat([1, 0], hrtfset.num_indices)))
gfb = GammatoneFilterbank(
            RestructureFilterbank(hrtfset_fb, cfN),
            tile(cf, hrtfset_fb.nchannels))
cochlea = FunctionFilterbank(gfb, lambda x:15*clip(x, 0, Inf)**(1.0/3.0))

eqs = '''
dV/dt = (I-V)/(1*ms)+0.1*xi/(0.5*ms)**.5 : 1
I : 1
'''
G = FilterbankGroup(cochlea, 'I', eqs, reset=0, threshold=1, refractory=5*ms)
cd = NeuronGroup(num_indices*cfN, eqs, reset=0, threshold=1, clock=G.clock)

C = Connection(G, cd, 'V')
for i in xrange(num_indices*cfN):
    C[i, i] = 0.5
    C[i+num_indices*cfN, i] = 0.5

M = SpikeMonitor(G, record=sound.duration<50*ms)
Mcd = SpikeMonitor(cd, record=sound.duration<50*ms)
counter = SpikeCounter(cd)

start = time.time()
run(sound.duration, report='stderr')
print 'Time taken:', time.time()-start

count = counter.count
count.shape = (num_indices, cfN)
count = count.T
count = reshape(count, (count.size,))

print 'Input spike rate:', float(M.nspikes)/(len(G)*G.clock.t)
print 'Coincidence spike rate:', float(Mcd.nspikes)/(len(cd)*G.clock.t)

if sound.duration<50*ms:
    subplot(221)
    raster_plot(M)
    subplot(223)
    raster_plot(Mcd)
    subplot(122)
ircam_plot_count(hrtfset, count, index=index)
show()
