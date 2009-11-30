from brian import *

clk_iz = Clock(dt=1*ms)
clk_iz2 = Clock(dt=.1*ms)
clk = Clock(dt=.01*ms)
method = 'RK'
Ninp = 10
rate = 50*Hz

a = 0.02/ms
b = 0.2/ms
c = -65*mV
d = 8*mV/ms
s = 6*mV

eqs = Equations('''
dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u : volt
du/dt = a*(b*v-u)                              : volt/second
''')
reset = '''
v = c
u += d
'''
threshold = 30*mV

G = NeuronGroup(1, eqs, threshold=threshold, reset=reset, clock=clk, method=method)
G_iz = NeuronGroup(1, eqs, threshold=threshold, reset=reset, clock=clk_iz)
G_iz2 = NeuronGroup(1, eqs, threshold=threshold, reset=reset, clock=clk_iz2)

# Izhikevich's numerical integration scheme
class Iz_state_updater(StateUpdater):
    def __init__(self, clk):
        self.clk = clk
    def __call__(self, G):
        G.v = G.v+0.5*self.clk.dt*((0.04/ms/mV)*G.v**2+(5/ms)*G.v+140*mV/ms-G.u)
        G.v = G.v+0.5*self.clk.dt*((0.04/ms/mV)*G.v**2+(5/ms)*G.v+140*mV/ms-G.u)
        G.u = G.u+self.clk.dt*a*(b*G.v-G.u)
G_iz._state_updater = Iz_state_updater(clk_iz)
G_iz2._state_updater = Iz_state_updater(clk_iz2)

G.v = G_iz.v = G_iz2.v = c
G.u = G_iz.u = G_iz2.u = b*c

inp = PoissonGroup(Ninp, rates=rate, clock=clk_iz)
C = Connection(inp, G, 'v', weight=s, sparseness=1)
C_iz = Connection(inp, G_iz, 'v', weight=s, sparseness=1)
C_iz2 = Connection(inp, G_iz2, 'v', weight=s, sparseness=1)

M = MultiStateMonitor(G, record=True, clock=clk_iz2)
M_iz = MultiStateMonitor(G_iz, record=True, clock=clk_iz)
M_iz2 = MultiStateMonitor(G_iz2, record=True, clock=clk_iz2)

run(1*second, report='stderr')

subplot(211)
plot(M['v'].times, M['v'][0], label='Brian dt='+str(clk.dt))
plot(M_iz['v'].times, M_iz['v'][0], label='Iz dt='+str(clk_iz.dt))
plot(M_iz2['v'].times, M_iz2['v'][0], label='Iz dt='+str(clk_iz2.dt))
legend(loc='upper right')
title('v')
subplot(212)
plot(M['u'].times, M['u'][0], label='Brian dt='+str(clk.dt))
plot(M_iz['u'].times, M_iz['u'][0], label='Iz dt='+str(clk_iz.dt))
plot(M_iz2['u'].times, M_iz2['u'][0], label='Iz dt='+str(clk_iz2.dt))
legend(loc='upper right')
title('u')
show()
