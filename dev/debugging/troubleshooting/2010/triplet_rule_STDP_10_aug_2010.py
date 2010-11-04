#
# Triplet rule STDP
#
# UNDER DEVELOPMENT - TESTING THE TRIPLET RULE IMPLEMENTATION
#
#
from brian import *

rate = 0.050 *kHz # Spike frequency
delta= 1 / rate # Interspike interval
N    = 60    # Number of spikes
t0   = 0 *ms # Time of the first spike
t1   = t0 + N * delta # Time of the last spike
tau  = 1*ms 

tau_plus  = 16.8 *ms
tau_x     = 101 *ms
tau_minus = 33.7 *ms
tau_y     = 125 *ms
A2_minus  = 7e-3
A3_minus  = 2.3e-4
A2_plus   = 5e-10
A3_plus   = 6.2e-3


presyn_train  = linspace(t0, t1, N, endpoint= False)
postsyn_train = presyn_train + 10*ms

input1 = SpikeGeneratorGroup(1, [(0, t * second) for t in presyn_train])
input2 = SpikeGeneratorGroup(1, [(0, t * second) for t in postsyn_train])

eqs = """
dv/dt = 0 * mV/ms  : mV
dw/dt = 0 /ms:1
"""

def myresetfunc(P, spikes):
 P.v[spikes] = H

SCR = SimpleCustomRefractoriness(myresetfunc, 0.01 *ms, state='v')

P = NeuronGroup(2, model=eqs, threshold="v > 5*mV", reset=0*mV)
P1 = P.subgroup(1)
P2 = P.subgroup(1)

C1 = Connection(input1, P1, 'v')
C1[0,0] = 10*mV
C2 = Connection(input2, P2, 'v')
C2[0,0] = 10*mV

C = Connection(P1, P2, 'w', weight=0)

eqs_stdp=Equations('''
dr1/dt = -r1/tau_plus  : 1
dr2/dt = -r2/tau_x     : 1
#www    = r1 - r2       : 1
do1/dt = -o1/tau_minus : 1
do2/dt = -o2/tau_y     : 1
#zzz    = o1 - o2       : 1
''')

#Triplet rule: fails to run
stdp=STDP(C,eqs=eqs_stdp,pre='r1+=1;w-=o1*(A2_minus+A3_minus*r2);r2+=1', post='o1+=1;w+=r1*(A2_plus+A3_plus*o2);o2+=1',wmax=100)#stdp=STDP(C,eqs=eqs_stdp,pre='r1+=1;w-=o1*(A2_minus+A3_minus*r2);r2+=1', post='o1+=1;w+=(A2_plus+A3_plus*o2);o2+=1',wmax=100)

#Modification: no mixing of pre and post terms on the weight update 
#stdp=STDP(C,eqs=eqs_stdp,pre='r1+=1;w-=o1*(A2_minus+A3_minus);r2+=1', post='o1+=1;w+=r1*(A2_plus+A3_plus);o2+=1',wmax=100)#stdp=STDP(C,eqs=eqs_stdp,pre='r1+=1;w-=o1*(A2_minus+A3_minus*r2);r2+=1', post='o1+=1;w+=(A2_plus+A3_plus*o2);o2+=1',wmax=100)


P.v = 0 *mV

M = StateMonitor(P, 'v', record=True)
run(600 *ms)
plot(M.times / ms, M[0] / mV, M.times / ms, M[1] / mV)
show()
