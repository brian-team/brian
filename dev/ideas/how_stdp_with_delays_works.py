from brian import *

N = 100
wmax = 0.1
tau_pre = 10 * ms
tau_post = 10 * ms
delta_A_pre = 0.1 * wmax
delta_A_post = -1.2 * delta_A_pre
max_delay = 5 * ms

G = NeuronGroup(N, 'dV/dt=(1.1-V)/(10*ms):1', reset=0, threshold=1)
G.V = rand(N)

C = Connection(G, G, 'V', sparseness=0.1, weight=lambda i, j:rand()*wmax,
               delay=(0 * ms, max_delay))

### STDP
G_pre = NeuronGroup(N, 'dA_pre/dt=-A_pre/tau_pre:1')
G_post = NeuronGroup(N, 'dA_post/dt=-A_post/tau_post:1')
A_pre_recent = RecentStateMonitor(G_pre, 'A_pre', duration=max_delay)
A_post_recent = RecentStateMonitor(G_post, 'A_post', duration=max_delay)
def update_on_pre_spikes_immediate(spikes):
    if len(spikes):
        G_pre.A_pre[spikes] += delta_A_pre
def update_on_pre_spikes_delayed(spikes):
    if len(spikes):
        times_seq = C.delayvec.get_rows(spikes)
        times_seq = [max_delay - times for times in times_seq]
        A_post_delayed_seq = A_post_recent.get_past_values_sequence(times_seq)
        for i, A_post_delayed in zip(spikes, A_post_delayed_seq):
            C.W[i, :] = clip(C.W[i, :] + A_post_delayed, 0, wmax)
def update_on_post_spikes(spikes):
    if len(spikes):
        G_post.A_post[spikes] += delta_A_post
        times_seq = C.delayvec.get_cols(spikes)
        A_pre_delayed_seq = A_pre_recent.get_past_values_sequence(times_seq)
        for i, A_pre_delayed in zip(spikes, A_pre_delayed_seq):
            C.W[:, i] = clip(C.W[:, i] + A_pre_delayed, 0, wmax)
M_pre_immediate = SpikeMonitor(G, function=update_on_pre_spikes_immediate)
M_pre_delayed = SpikeMonitor(G, function=update_on_pre_spikes_delayed, delay=max_delay)
M_post = SpikeMonitor(G, function=update_on_post_spikes)


M = SpikeMonitor(G)
run(1 * second)
raster_plot(M)
show()
