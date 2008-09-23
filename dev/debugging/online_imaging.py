# Online imaging of network activity
# Not working great yet
from brian import *

tau=10*ms
tauw=100*ms
v0=.5
sigma=.5
N=100

eqs='''
dv/dt=(v0-v)/tau + sigma*xi/tau**.5: 1
dw/dt=-w/tauw : 1
'''
def myreset(P,spikes): # idea: maybe P should be a subgroup?
    P.v_[spikes]=0
    P.w_[spikes]+=1
group=NeuronGroup(N,model=eqs,threshold=1,reset=myreset)
C=Connection(group,group,'v')
C.connect_full(group,group,weight=lambda i,j:.05*exp(-abs(i-j)*.1))
group.v=rand(N)

figure()
show()

@network_operation(clock=Clock(dt=100*ms))
def op():
    imshow(array(dot(ones((N,1)),group.w_.reshape((1,N)))))
    
run(1000*ms)
