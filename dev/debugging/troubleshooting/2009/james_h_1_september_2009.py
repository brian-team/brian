from brian import *

gmax=1.5
tau_pre=20*ms
tau_post=tau_pre
dA_pre=.001
dA_post=-dA_pre*tau_pre/tau_post*1.05

input=PoissonGroup(2, 50*Hz)
output=NeuronGroup(2, 'V:1',
                     reset=0, threshold=0.5)
relay=IdentityConnection(input, output, delay=1*ms)
C=Connection(input, output)
C[0, 0]=C[1, 1]=1e-100

eqs_stdp='''
dA_pre/dt = -A_pre/tau_pre : 1
dA_post/dt = -A_post/tau_post : 1
'''
dA_post*=gmax
dA_pre*=gmax
class Heap(object):
    pass
h=Heap()
h.dA_post=dA_post
h.dA_pre=dA_pre
stdp=STDP(C , eqs=eqs_stdp, pre='A_pre+=h.dA_pre; w+=A_post',
          post='A_post+=h.dA_post; w+=A_pre', wmax=gmax)

Mi=SpikeMonitor(input)
Mo=SpikeMonitor(output)

w0=[]
w1=[]
@network_operation(clock=EventClock(dt=1*ms))
def recweights():
    w0.append(C[0, 0])
    w1.append(C[1, 1])

run(1*second)
h.dA_post*=5
h.dA_pre*=5
run(1*second)
plot(w0)
plot(w1)
show()
