import cloud
import time
from brian import *
from multimap import *

def stdp_example(N = 1000):
    taum=10*ms
    tau_pre=20*ms
    tau_post=tau_pre
    Ee=0*mV
    vt=-54*mV
    vr=-60*mV
    El=-74*mV
    taue=5*ms
    F=15*Hz
    gmax=.01
    dA_pre=.01
    dA_post=-dA_pre*tau_pre/tau_post*1.05
    
    eqs_neurons='''
    dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
    dge/dt=-ge/taue : 1
    '''
    
    input=PoissonGroup(N,rates=F)
    neurons=NeuronGroup(1,model=eqs_neurons,threshold=vt,reset=vr)
    synapses=Connection(input,neurons,'ge',weight=rand(len(input),len(neurons))*gmax)
    neurons.v=vr
    
    #stdp=ExponentialSTDP(synapses,tau_pre,tau_post,dA_pre,dA_post,wmax=gmax)
    ## Explicit STDP rule
    eqs_stdp='''
    dA_pre/dt=-A_pre/tau_pre : 1
    dA_post/dt=-A_post/tau_post : 1
    '''
    dA_post*=gmax
    dA_pre*=gmax
    stdp=STDP(synapses,eqs=eqs_stdp,pre='A_pre+=dA_pre;w+=A_post',
              post='A_post+=dA_post;w+=A_pre',wmax=gmax)
    
    rate=PopulationRateMonitor(neurons)
    
    run(10*second,report='text')
    
    return mean(rate.rate)

ilist = [1000 for _ in xrange(32)]
t1 = time.clock()
jids = multimap(stdp_example, ilist)
t = time.clock()-t1
results = retrieve(jids)

print "All jobs done in %.1f seconds, results :" % t
print results

