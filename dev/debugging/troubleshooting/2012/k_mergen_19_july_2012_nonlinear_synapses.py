"""
NMDA synapses
"""
from brian import NeuronGroup, StateMonitor, set_global_preferences, run, Clock
from brian.stdunits import ms
import time
from brian.experimental.synapses import *

from brian import log_level_debug
log_level_debug()

set_global_preferences(useweave=True,usecodegen=True,usecodegenweave=True,usenewpropagate=True,usecstdp=True)

from matplotlib.pyplot import plot, show, subplot

params = {}
params["t_Nr"] = 2*ms
params["t_Nf"] = 80*ms
params["t_AMPA"] = 5*ms


simclock = Clock(dt=0.01*ms)

input=NeuronGroup(2,model='dv/dt=1/(10*ms):1', threshold=1, reset=0,clock=simclock)
neurons = NeuronGroup(1, model="""dv/dt=(NMDAo+AMPAo-v)/(10*ms) : 1
                                  NMDAo : 1
                                  AMPAo : 1""", freeze = True,clock=simclock)

ampadyn = '''
        dAMPAoS/dt = -AMPAoS/t_AMPA            : 1
        AMPAi = AMPAoS
        AMPAo = AMPAoS / (t_AMPA /msecond)     : 1
        '''

nmdadyn = '''
        dNMDAoS/dt = (1/t_Nr)*(Nnor*NMDAi-NMDAoS)                                         : 1
        dNMDAi/dt = -(1/t_Nf)*NMDAi                                                       : 1 
        Nnor = (t_Nf/t_Nr)**((t_Nr)/(t_Nf - t_Nr))                                        : 1
        Nscal = (t_Nf/msecond)**(t_Nf/(t_Nf - t_Nr))/(t_Nr/msecond)**(t_Nr/(t_Nf - t_Nr)) : 1
        NMDAo = NMDAoS / Nscal                                                            : 1
        w : 1 # synaptic weight
        '''

s_eq = SynapticEquations(nmdadyn+ampadyn,**params)

S=Synapses(input,neurons,
           model=s_eq,
           pre='NMDAi+=w\nAMPAi+=w',post='',clock=simclock) # NMDA synapses
neurons.NMDAo=S.NMDAo
neurons.AMPAo=S.AMPAo
S[:,:]=True
S.w=[0.01,0.01]
S.delay=[0.003,0.005]
input.v=[0.,0.5]

M=StateMonitor(S,'NMDAo',record=True,clock=simclock)
M0 = StateMonitor(S,'NMDAi',record=True,clock=simclock)
M2=StateMonitor(S,'AMPAo',record=True,clock=simclock)
Mn=StateMonitor(neurons,'v',record=True,clock=simclock)

run(100*ms)
subplot(411)
plot(M0.times/ms,M0[0])
plot(M0.times/ms,M0[1])
subplot(412)
plot(M.times/ms,M[0])
plot(M.times/ms,M[1])
subplot(413)
plot(M2.times/ms,M2[0])
plot(M2.times/ms,M2[1])

subplot(414)
plot(Mn.times/ms,Mn[0])
show()
