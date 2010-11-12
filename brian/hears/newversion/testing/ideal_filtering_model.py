from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *
import time

hrtfdb = IRCAM_LISTEN(r'D:\HRTF\IRCAM')
subject = 1002
hrtfset = hrtfdb.load_subject(subject)
index = randint(hrtfset.num_indices)
cfmin, cfmax, cfN = 150*Hz, 5*kHz, 40

sound = Sound.whitenoise(500*ms)

def ircam_plot_count(hrtfset, count, index=None, showbest=True, absolute=False,
                     vmin=None, vmax=None, I=None, ms=20, mew=2, indexcol='k', bestcol='w'):
    if I is None: I = arange(len(count))
    count = array(count, dtype=float)
    num_indices = hrtfset.num_indices
    count.shape = (count.size/num_indices, num_indices)
    count = sum(count, axis=0)
    img = zeros((10, 24))
    for i, c in enumerate(count):
        if i in I:
            elev = hrtfset.elev[i]
            azim = hrtfset.azim[i]
            if elev<60:
                w = 1
            elif elev==60:
                w = 2
            elif elev==75:
                w = 4
            elif elev==90:
                w = 24
                azim = -180
            if azim>=180: azim -= 360
            x = int((azim+180)/15)
            y = int((elev+45)/15)
            img[y, x:x+w] = c
    if absolute:
        imshow(img, origin='lower left', interpolation='nearest', extent=(-180-7.5, 180-7.5, -45-7.5, 90+7.5),
               vmin=vmin, vmax=vmax)
        axis('tight')
    else:
        imshow(img, origin='lower left', interpolation='nearest', extent=(-180-7.5, 180-7.5, -45-7.5, 90+7.5))
        axis('tight')
    if index is not None:
        azim = hrtfset.azim[index]
        elev = hrtfset.elev[index]
        if azim>=180: azim -= 360
        plot([azim], [elev], '+', ms=ms, mew=mew, color=indexcol)
    if showbest:
        i = argmax(count)
        azim = hrtfset.azim[i]
        elev = hrtfset.elev[i]
        if azim>=180: azim -= 360
        plot([azim], [elev], 'x', ms=ms, mew=mew, color=bestcol)
    return img        

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

R = 15.0

cochlea = FunctionFilterbank(gfb, lambda x:R*clip(x, 0, Inf)**(1.0/3.0))

eqs = '''
dV/dt = (I-V)/(1*ms)+0.1*xi/(0.5*ms)**.5 : 1
I : 1
'''

G = FilterbankGroup(cochlea, 'I', eqs, reset=0, threshold=1, refractory=5*ms)

cd = NeuronGroup(num_indices*cfN, eqs, reset=0, threshold=1)

C = Connection(G, cd, 'V')
C[arange(num_indices*cfN), arange(num_indices*cfN)] = 0.5
C[arange(num_indices*cfN)+num_indices*cfN, arange(num_indices*cfN)] = 0.5

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
