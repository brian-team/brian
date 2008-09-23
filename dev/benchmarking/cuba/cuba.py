from brian import *
import time
import pickle
from operator import itemgetter
from cuba_runopts import *

def cubanetwork(N, Nsyn=80., we=1.62*mV):
    eqs='''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''
    
    Ne = int(N*0.8)
    Ni = N-Ne
    
    P=NeuronGroup(N,model=eqs,
                  threshold=-50*mV,reset=-60*mV, refractory=5*ms)
    P.v=-60*mV+10*mV*rand(len(P))
    Pe=P.subgroup(Ne)
    Pi=P.subgroup(Ni)
    
    if cuba_opts['connections']:
        Ce=Connection(Pe,P,'ge')
        Ci=Connection(Pi,P,'gi')    
        cp = Nsyn/N    
        Ce.connect_random(Pe, P, cp, weight=we)
        Ci.connect_random(Pi, P, cp, weight=-9*mV)
#        from brian.experimental.stdp_sparse import SparseSTDPConnectionMatrix
#        Ce.W = SparseSTDPConnectionMatrix(Ce.W)
#        Ci.W = SparseSTDPConnectionMatrix(Ci.W)
    else:
        Ce = None
        Ci = None
    
    M = PopulationSpikeCounter(P)
    
    if cuba_opts['connections']:
        net = Network(P, Ce, Ci, M)
    else:
        net = Network(P, M)
    net.run(0*ms)    
    net.Nsyn = Nsyn
    net.we = we
    
    return P, Pe, Pi, Ce, Ci, M, net

def cuba(P, Pe, Pi, Ce, Ci, M, net):
    reinit_default_clock()
    P.v=-60*mV+10*mV*rand(len(P))
    if cuba_opts['connections']:
        Ce.__init__(Pe,P,'ge')
        Ci.__init__(Pi,P,'gi')
        cp = net.Nsyn/len(P)
        Ce.connect_random(Pe, P, cp, weight=net.we)
        Ci.connect_random(Pi, P, cp, weight=-9*mV)
        Ce.W.freeze()
        Ci.W.freeze()
    M.reinit()
    tstart = time.time()
    net.run(duration)
    tend = time.time()
    print '.',
    return tend-tstart, M.nspikes

def cuba_average(N, repeats, best, Nsyn=80., we=1.62*mV):
    cubadata = cubanetwork(N, Nsyn, we)
    cubaruns = [cuba(*cubadata) for i in range(repeats)]
    print
    print 'Finished run', N, '-', Nsyn, '-', we
    cubaruns.sort(key=itemgetter(0))
    cubaruns = cubaruns[:best]
    t, ns = zip(*cubaruns)
    t = array(t)
    ns = array(ns)
    return N, mean(t), mean(ns), cubaruns 

def cuba_runs(Nvals, repeats, best):
    return [cuba_average(N, repeats, best) for N in Nvals]

def cuba_runs_varyconnectivity(N, Nsynvals, repeats, best):
    return [cuba_average(N, repeats, best, Nsyn) for Nsyn in Nsynvals]

def cuba_runs_varywe(N, wevals, repeats, best):
    return [cuba_average(N, repeats, best, we=we) for we in wevals]

def do_runs(fname='data/brian_cuba_results.pkl'):
    ca = cuba_runs(Nvals, repeats, best)
    output = open(fname, 'wb')
    pickle.dump(ca,output,-1)
    output.close()
    return ca

def do_runs_varyconnectivity(fname='data/brian_cuba_varycon_results.pkl'):
    ca = cuba_runs_varyconnectivity(N_varycon, Nsynvals, repeats, best)
    output = open(fname, 'wb')
    pickle.dump(ca,output,-1)
    output.close()
    return ca

def do_runs_varywe(fname='data/brian_cuba_varywe_results.pkl'):
    ca = cuba_runs_varywe(N_varywe, wevals, repeats, best)
    output = open(fname, 'wb')
    pickle.dump(ca,output,-1)
    output.close()
    return ca

if __name__=='__main__':
    #print do_runs()
    #print do_runs_varyconnectivity()
    X = cubanetwork(16000,we=9*mV)
    print 'Built X'
    print cuba(*X)