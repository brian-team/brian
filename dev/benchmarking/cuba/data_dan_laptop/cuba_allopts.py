import brian_no_units
from brian import *
import time
import pickle
from operator import itemgetter
from cuba_runopts import *

set_global_preferences(useweave=True)

def cubanetwork(N):
    Vr = -49 * mV
    invtauv = 1 / (20 * ms)
    invtauge = 1 / (5 * ms)
    invtaugi = 1 / (10 * ms)
    eqs = '''
    dv/dt = (ge+gi-(v-Vr))*invtauv : volt
    dge/dt = -ge*invtauge : volt
    dgi/dt = -gi*invtaugi : volt
    '''

    Ne = int(N * 0.8)
    Ni = N - Ne

    P = NeuronGroup(N, model=eqs,
                  threshold= -50 * mV, reset= -60 * mV,
                  compile=True, freeze=True)
    P.v = -60 * mV + 10 * mV * rand(len(P))
    Pe = P.subgroup(Ne)
    Pi = P.subgroup(Ni)

    Ce = Connection(Pe, P, 'ge')
    Ci = Connection(Pi, P, 'gi')
    Ce.connect_random(Pe, P, 0.02, weight=1.62 * mV)
    Ci.connect_random(Pi, P, 0.02, weight= -9 * mV)

    net = Network(P, Ce, Ci)
    net.run(0 * ms)

    return P, Pe, Pi, Ce, Ci, net

def cuba(P, Pe, Pi, Ce, Ci, net):
    reinit_default_clock()
    P.v = -60 * mV + 10 * mV * rand(len(P))
    Ce.__init__(Pe, P, 'ge')
    Ci.__init__(Pi, P, 'gi')
    Ce.connect_random(Pe, P, 0.02, weight=1.62 * mV)
    Ci.connect_random(Pi, P, 0.02, weight= -9 * mV)
    tstart = time.time()
    net.run(duration)
    tend = time.time()
    return tend - tstart, 0

def cuba_average(N, repeats, best):
    cubadata = cubanetwork(N)
    cubaruns = [cuba(*cubadata) for i in range(repeats)]
    print 'Finished run', N
    cubaruns.sort(key=itemgetter(0))
    cubaruns = cubaruns[:best]
    t, ns = zip(*cubaruns)
    t = array(t)
    ns = array(ns)
    return N, mean(t), mean(ns), cubaruns

def cuba_runs(Nvals, repeats, best):
    return [cuba_average(N, repeats, best) for N in Nvals]

def do_runs(fname='brian_cuba_results_allopts.pkl'):
    ca = cuba_runs(Nvals, repeats, best)
    output = open(fname, 'wb')
    pickle.dump(ca, output, -1)
    output.close()
    return ca

if __name__ == '__main__':
    print do_runs()
#    cubadata = cubanetwork(32000)
#    print cuba(*cubadata)
#    def f():
#        cubadata = cubanetwork(4000)
#        print cuba(*cubadata)
#    f()
#    import cProfile as profile
#    profile.run('f()')
#    import hotshot, hotshot.stats
#    prof = hotshot.Profile('cuba_allopts.prof')
#    prof.runcall(f)
#    stats = hotshot.stats.load("cuba_allopts.prof")
#    stats.strip_dirs()
#    stats.sort_stats('cumulative','time','calls')
#    stats.print_stats()
