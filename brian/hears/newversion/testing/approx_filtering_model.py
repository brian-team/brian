from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *
from plot_count import ircam_plot_count
import time
import pickle

hrtfdb = IRCAM_LISTEN(r'F:\HRTF\IRCAM')
subject = 1002
hrtfset = hrtfdb.load_subject(subject)
index = randint(hrtfset.num_indices)
cfmin, cfmax, cfN = 150*Hz, 5*kHz, 40
sound = Sound.whitenoise(500*ms)

hrtf = hrtfset.hrtf[index]
cf = erbspace(cfmin, cfmax, cfN)
num_indices = hrtfset.num_indices

try:
    all_itds, all_ilds = pickle.load(open('itd_ild_data.pkl', 'r'))
    print 'Loaded pregenerated ITD/ILD data'
except IOError:
    print 'Performing ITD/ILD analysis (may take several minutes)'
    start = time.time()
    all_itds = []
    all_ilds = []
    for j in xrange(num_indices):
        hrirset = Sound(hrtfset.hrtf[j].fir.T, samplerate=hrtfset.samplerate)
        fb = GammatoneFilterbank(RestructureFilterbank(hrirset, cfN),
                                 hstack((cf, cf)))
        fb.buffer_init()
        filtered_hrirset = fb.buffer_fetch(0, hrtfset.num_samples)
        itds = []
        ilds = []
        for i in xrange(cfN):
            left = filtered_hrirset[:, i]
            right = filtered_hrirset[:, i+cfN]
            Lf = fft(hstack((left, zeros(len(left)))))
            Rf = fft(hstack((right[::-1], zeros(len(right)))))
            C = ifft(Lf*Rf).real
            i = argmax(C)+1-len(left)
            itds.append(i/hrtfset.samplerate)
            ilds.append(sqrt(amax(C)/sum(right**2)))
        itds = array(itds)
        ilds = array(ilds)
        all_itds.append(itds)
        all_ilds.append(ilds)
    print 'Time taken:', time.time()-start
    pickle.dump((all_itds, all_ilds), open('itd_ild_data.pkl', 'w'))
    print 'File saved as itd_ild_data.pkl'

d = array([all_itds[j][i] for i in xrange(cfN) for j in xrange(num_indices)])
g = array([all_ilds[j][i] for i in xrange(cfN) for j in xrange(num_indices)])
gains = hstack((1/g, g))
gains_dB = 20*log10(gains)
abs_gains_dB = abs(gains_dB)
r = -abs_gains_dB[:len(gains)/2]
r = hstack((r, r))
gains_dB += r
gains = 10**(gains_dB/20)
gains = reshape(gains, (1, len(gains)))
delays_L = where(d>=0, zeros(len(d)), -d)
delays_R = where(d>=0, d, zeros(len(d)))
delay_max = max(amax(delays_L), amax(delays_R))*second

hrtf_fb = hrtf.filterbank(sound)
gfb = GammatoneFilterbank(
        RestructureFilterbank(hrtf_fb, cfN),
        hstack((cf, cf)))
gains_fb = FunctionFilterbank(
        RestructureFilterbank(gfb, num_indices),
        lambda x:x*gains,
        )
cochlea = FunctionFilterbank(gains_fb, lambda x:15*clip(x, 0, Inf)**(1.0/3.0))

eqs = '''
dV/dt = (I-V)/(1*ms)+0.1*xi/(0.5*ms)**.5 : 1
I : 1
'''
G = FilterbankGroup(cochlea, 'I', eqs, reset=0, threshold=1, refractory=5*ms)
cd = NeuronGroup(num_indices*cfN, eqs, reset=0, threshold=1, clock=G.clock)

C = Connection(G, cd, 'V', delay=True, max_delay=delay_max)
for i in xrange(num_indices*cfN):
    C[i, i] = 0.5
    C[i+num_indices*cfN, i] = 0.5
    C.delay[i, i] = delays_L[i]
    C.delay[i+cfN*num_indices, i] = delays_R[i]

M = SpikeMonitor(G, record=sound.duration<50*ms)
Mcd = SpikeMonitor(cd, record=sound.duration<50*ms)
counter = SpikeCounter(cd)

start = time.time()
run(sound.duration, report='stderr')
print 'Time taken:', time.time()-start

count = counter.count

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
