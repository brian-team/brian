from brian import *
import time

N = 100
N_set = [int(N) for N in randint(100,8000,size=100)]
#set_global_preferences(useweave=True)

us = usecond
ns = nsecond

def get_predicted_time(N, r):
    theta = 0.02*N
    T = 1*second
    dt = 0.1*ms
    if get_global_preference('useweave'):
        pass
    else:
        overhead_fudge = 80*us
        wN = 6.4*us
        wC = 16.4*us
        wG = 22.3*us
        aG = 45*ns
        aG_cachemiss = 138*ns
        athr = 35.1*ns
        athr_cachemiss = 90*ns
        asp = 11*us
        asyn = 59.3*ns
    p_lower = (T/dt)*(overhead_fudge+wN+2*wC+wG+N*(aG+athr))+T*r*N*(asp+asyn*theta)
    p_middle = (T/dt)*(overhead_fudge+wN+2*wC+wG+N*(aG_cachemiss+athr))+T*r*N*(asp+asyn*theta)
    p_upper = (T/dt)*(overhead_fudge+wN+2*wC+wG+N*(aG_cachemiss+athr_cachemiss))+T*r*N*(asp+asyn*theta)
    return (p_lower, p_middle, p_upper)

def get_exp(N):
    eqs='''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''
    Ne = int(N*0.8)
    Ni = N-Ne
    P=NeuronGroup(N,model=eqs,
                  threshold=-50*mV,reset=-60*mV)
    P.v=-60*mV+10*mV*rand(len(P))
    Pe=P.subgroup(Ne)
    Pi=P.subgroup(Ni)
    Ce=Connection(Pe,P,'ge')
    Ci=Connection(Pi,P,'gi')
    Ce.connect_random(Pe, P, 0.02,weight=1.62*mV)
    Ci.connect_random(Pi, P, 0.02,weight=-9*mV)
    M = PopulationSpikeCounter(P)
    net = Network(P, Ce, Ci, M)
    net.run(100*ms)
    M.reinit()
    t = time.time()
    net.run(1*second)
    t = time.time()-t
    return (float(M.nspikes)/(N*1.*second), t*second)

#r, t = get_exp(N)
#print 'Rate:', r
#print 'Actual time:', t
#print 'Predicted time:', get_predicted_time(N, r)

rates, t = zip(*[get_exp(N) for N in N_set])
p, pcm_middle, pcm_max = zip(*[get_predicted_time(N,r) for N,r in zip(N_set,rates)])
subplot(121)
plot(N_set, t, '.')
plot(N_set, p, '.')
plot(N_set, pcm_middle, '.')
plot(N_set, pcm_max, '.')
legend(('Actual','No cache miss','Cache miss on group','All cache misses'),'upper left')
subplot(122)
plot(N_set, rates, '.')
show()