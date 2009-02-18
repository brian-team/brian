from brian import *
def howmanyspikes(excitatory_weight):
    from brian import hold, ma, ms, NeuronGroup, rand, io, any, threshold, reset, eig, mod, mV, eigh, connect, x, pi, SpikeMonitor, run, e, Connection, random, volt, f, ion
    eqs='''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''
    P=NeuronGroup(4000,model=eqs,threshold=-50*mV,reset=-60*mV)
    P.v=-60*mV+10*mV*rand(len(P))
    Pe=P.subgroup(3200)
    Pi=P.subgroup(800)
    Ce=Connection(Pe,P,'ge')
    Ci=Connection(Pi,P,'gi')
    Ce.connect_random(Pe, P, 0.02,weight=excitatory_weight)
    Ci.connect_random(Pi, P, 0.02,weight=-9*mV)
    M=SpikeMonitor(P)
    run(100*ms)
    return M.nspikes
