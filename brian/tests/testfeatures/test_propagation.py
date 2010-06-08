from brian import *

def test_structures():
    reinit_default_clock()
    H = NeuronGroup(1, 'V:1\nmod:1', reset=0, threshold=1)
    H.V = 2
    H.mod = 1
    G = [NeuronGroup(10, 'V:1', reset=0, threshold=1) for _ in range(12)]
    M = [SpikeMonitor(g) for g in G]
    C0 = Connection(H, G[0], 'V', weight=2)
    C1 = Connection(H, G[1], 'V', weight=2, structure='dense')
    C2 = Connection(H, G[2], 'V', weight=2, structure='dynamic')
    C3 = Connection(H, G[3], 'V', weight=2, modulation='mod')
    C4 = Connection(H, G[4], 'V', weight=2, structure='dense', modulation='mod')
    C5 = Connection(H, G[5], 'V', weight=2, structure='dynamic', modulation='mod')
    C6 = Connection(H, G[6], 'V', weight=2, delay=True)
    C7 = Connection(H, G[7], 'V', weight=2, delay=True, structure='dense')
    C8 = Connection(H, G[8], 'V', weight=2, delay=True, structure='dynamic')
    C9 = Connection(H, G[9], 'V', weight=2, delay=True, modulation='mod')
    C10 = Connection(H, G[10], 'V', weight=2, delay=True, structure='dense', modulation='mod')
    C11 = Connection(H, G[11], 'V', weight=2, delay=True, structure='dynamic', modulation='mod')
    for c in [C6, C7, C8, C9, C10, C11]:
        c.delay[0, :] = arange(10) * defaultclock.dt + defaultclock.dt / 2
    run(2 * ms)
    for k, m in enumerate(M):
        assert len(m.spikes), 'Problem with connection ' + str(k)
        i, j = zip(*m.spikes)
        i = array(i)
        j = array(j)
        j = array((j + defaultclock.dt / 2) / defaultclock.dt, dtype=int)
        assert (i == arange(10)).all(), 'Problem with connection ' + str(k)
        if k < 6:
            assert (j == 1).all(), 'Problem with connection ' + str(k)
        else:
            assert (j == (1 + arange(10))).all(), 'Problem with connection ' + str(k) + ': j=' + str(j)

if __name__ == '__main__':
    test_structures()
