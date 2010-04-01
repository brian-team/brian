import cloud
import time
from brian import *

api_keys = ['1307',
            '1308']

api_secretkeys = ['9106bed9b15b00197df2734102a66a9ce5698f1d',
                  '69d0919b47fab35e959bd7762c163bd4826a393c']

n = 16 # number of parallel machines for each account

def multimap(fun, args):
    # jids[i] contains the job indices for account i
    jids = [None for _ in xrange(len(api_keys))]
    
    # Launches the jobs
    k = 0
    for i in xrange(len(api_keys)):
        api_key = api_keys[i]
        api_secretkey = api_secretkeys[i]
        
        args_tmp = args[k:k+n]
        if len(args_tmp)>0:
            print "Launching %d jobs with account %d..." % (len(args_tmp), i+1)
            cloud.setkey(api_key=api_key, api_secretkey=api_secretkey)
            jids[i] = cloud.map(fun, args_tmp)
            k += n

    results = []
    # Retrieves the results
    for i in xrange(len(api_keys)):
        api_key = api_keys[i]
        api_secretkey = api_secretkeys[i]
        cloud.setkey(api_key=api_key, api_secretkey=api_secretkey)
        print "Retrieving results for account %d..." % (i+1)
        results.extend(cloud.result(jids[i]))
        
    return results

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


t1 = time.clock()
ilist = [1000 for _ in xrange(32)]
results = multimap(stdp_example, ilist)

print "All jobs done in %.1f seconds, results :" % (time.clock()-t1)
print results
