'''
Comparison of two algorithms for the new spike/refractoriness mechanism. These
algorithms are capable of arbitrarily long refractory periods which don't need
to be specified in advance, can change from moment to moment (although once a
spike is fired the refractory period for that firing is fixed), and can signal
differently for each condition (1) spike fired, (2) in refractory period,
(3) leaving refractory period. These could be used by the refractoriness
mechanism in different ways.

Results:

N = 100, duration = 10000, thr=0.5, refracperiod = arange(N)
algo 1: .7
algo 2: .96

N = 100, duration = 10000, thr=0.5, refracperiod = zeros(N)
algo 1: .72
algo 2: .97

N = 100, duration = 10000, thr=0.5, refracperiod = 10*ones(N)
algo 1: .75
algo 2: .99

N = 100, duration = 10000, thr=0.5, refracperiod = 500*ones(N)
algo 1: .7
algo 2: .95

So it looks like distribution of refrac periods doesn't count much.

N = 100, duration = 10000, thr=0.1, refracperiod = arange(N)
algo 1: .69
algo 2: .91

N = 100, duration = 10000, thr=0.01, refracperiod = arange(N)
algo 1: .67
algo 2: .875

N = 100, duration = 10000, thr=0.9, refracperiod = arange(N)
algo 1: .72
algo 2: .97

So it looks like firing rate doesn't matter much.

N = 1000, duration = 1000, thr=0.5, refracperiod = arange(N)
algo 1: .19
algo 2: .24

N = 10000, duration = 10000, thr=0.5, refracperiod = arange(N)
algo 1: 14.3
algo 2: 16.5

So it looks like number of neurons doesn't make much difference.

Conclusion: algo 1 > algo 2

For large N, it appears to take about as long to do refractoriness as it does
to do thresholding. I think this seems relatively inexpensive.
'''

from brian import *
import time

N = 10000
duration = 10000
thr = 0.5
dorecord = False
algo = 1

refracperiod = zeros(N, dtype=int)#arange(N)

staterec = []
active = ones(N, dtype=bool)

start = time.time()
for t in xrange(duration):
    threshold = (rand(N)<thr).nonzero()[0]
    spikes = threshold[active[threshold]]
end = time.time()

t_thresh = end-start
print 'Time spent doing thresholding:', t_thresh

if algo==1:

    refend = zeros(N, dtype=int)
    ref = zeros(N, dtype=int)
    numref = 0

    start = time.time()
    for t in xrange(duration):
        if dorecord: state = zeros(N)
        threshold = (rand(N)<thr).nonzero()[0]
        spikes = threshold[active[threshold]]
        nspikes = len(spikes)
        active[spikes] = False
        ref[numref:numref+nspikes] = spikes
        refend[numref:numref+nspikes] = t+refracperiod[spikes]
        if dorecord: state[ref[:numref]] = 2
        numref += nspikes
        # sorted(ref[:numref]) # at moderate cost (an additional 30%) can return sorted array, which might be better for cache?
        # on the other hand, cache may not be so important unless a large number of neurons are refractory. Probably better to leave
        # this out overall.
        if dorecord: state[spikes] = 1
        I = (t>=refend[:numref])
        notI = -I
        leaving = ref[I.nonzero()[0]]
        staying = notI.nonzero()[0]
        active[leaving] = True
        if dorecord: state[leaving] = 3
        nstaying = len(staying)
        ref[:nstaying] = ref[staying]
        refend[:nstaying] = refend[staying]
        numref = nstaying
        if dorecord: state += active*0.3
        if dorecord: staterec.append(state)
    end = time.time()

elif algo==2:

    refend = zeros(N, dtype=int)
    ref = zeros(0, dtype=int)
    numref = 0

    start = time.time()
    for t in xrange(duration):
        if dorecord: state = zeros(N)
        threshold = (rand(N)<thr).nonzero()[0]
        spikes = threshold[active[threshold]]
        if dorecord: state[ref] = 2
        ref = union1d(ref, spikes) # this involves doing a sort operation, although it could be accelerated with weave easily enough
        active[spikes] = False
        refend[spikes] = t+refracperiod[spikes]
        if dorecord: state[spikes] = 1
        I = (t>=refend[ref])
        notI = -I
        leaving = ref[I]
        active[leaving] = True
        if dorecord: state[leaving] = 3
        ref = ref[notI]
        if dorecord: state += active*0.3
        if dorecord: staterec.append(state)
    end = time.time()
    
print 'Time spent total:', end-start
print 'Time spent on algorithm:', end-start-t_thresh

if dorecord:
    imshow(array(staterec).T, interpolation='nearest', origin='bottom left', aspect='auto')
    colorbar()
    show()
