from brian import *

set_global_preferences(useweave=True)

inp=PoissonGroup(1, rates=30*Hz)

###### Standard ODE approach
eqs='''
    dV/dt = -V/(0.5*second) : volt
    '''
G=NeuronGroup(1, model=eqs, threshold=10*mV, reset=0*mV)
C=Connection(inp, G)
C.connect_full(inp, G, weight=2*mV)
M=StateMonitor(G, 'V', record=True)

###### Event driven approach, maybe faster if spikes are rare
eqs_ed='''
    V : volt
    exc : volt
    '''
G_ed=NeuronGroup(1, model=eqs_ed, threshold=10*mV, reset=0*mV)
C_ed=Connection(inp, G_ed, 'exc')
C_ed.connect_full(inp, G_ed, weight=2*mV)
M_ed=StateMonitor(G_ed, 'V', record=True)
G_ed.last_event_time=0.0
@network_operation(when='before_groups')
def update_G_ed():
    if G_ed.exc_[0]>1e-10:
        t=defaultclock._t
        G_ed.V_[0]=G_ed.V_[0]*exp((G_ed.last_event_time-t)/0.5)+G_ed.exc_[0]
        G_ed.exc_[0]=0.0
        G_ed.last_event_time=t

####### Run and plot
run(1*second)
plot(M.times, M[0])
plot(M_ed.times, M_ed[0])
show()
