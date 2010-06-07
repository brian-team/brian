from brian import *

#tau_pre = 5*ms
tau_post=5*ms
gmax=1.0
delta_A_pre=0.2
delta_A_post=-0.3

G=NeuronGroup(2, 'V:1', reset=0, threshold=1)

C=Connection(G, G, 'V')

C[0, 1]=0.5

stdp=STDP(C, '''
            dA_pre/dt  = -0.1*sqrt(clip(A_pre,0,inf))/tau_pre   : 1
            dA_post/dt = -A_post/tau_post : 1
            tau_pre : second
            ''', pre='''
            A_pre += delta_A_pre
            w += A_post
            ''', post='''
            A_post += delta_A_post
            w += A_pre
            ''', wmax=gmax)

stdp.tau_pre=[2*ms, .1*ms]

M_pre=StateMonitor(stdp.pre_group, 'A_pre', record=True)
M_post=StateMonitor(stdp.post_group, 'A_post', record=True)

wvals=[]

@network_operation
def rec_wvals():
    wvals.append(C[0, 1])

run(1*ms)
G.V[1]=2
run(1*ms)
G.V[0]=2
run(1*ms)

subplot(221)
M_pre.plot()
subplot(222)
M_post.plot()
subplot(223)
plot(wvals)

show()
