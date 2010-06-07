from brian import *

N=100
wmax=0.1
tau_pre=10*ms
tau_post=10*ms
delta_A_pre=0.1*wmax
delta_A_post=-1.2*delta_A_pre

G=NeuronGroup(N, 'dV/dt=(1.1-V)/(10*ms):1', reset=0, threshold=1)
G.V=rand(N)

C=Connection(G, G, 'V', sparseness=0.1, weight=lambda i, j:rand()*wmax)

### STDP
G_pre=NeuronGroup(N, 'dA_pre/dt=-A_pre/tau_pre:1')
G_post=NeuronGroup(N, 'dA_post/dt=-A_post/tau_post:1')
def update_on_pre_spikes(spikes):
    if len(spikes):
        G_pre.A_pre[spikes]+=delta_A_pre
        for i in spikes:
            C.W[i, :]=clip(C.W[i, :]+G_post.A_post, 0, wmax)
def update_on_post_spikes(spikes):
    if len(spikes):
        G_post.A_post[spikes]+=delta_A_post
        for i in spikes:
            C.W[:, i]=clip(C.W[:, i]+G_pre.A_pre, 0, wmax)
M_pre=SpikeMonitor(G, function=update_on_pre_spikes)
M_post=SpikeMonitor(G, function=update_on_post_spikes)


M=SpikeMonitor(G)
run(1*second)
raster_plot(M)
show()
