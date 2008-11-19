from brian import *
import brianlib as bl

eqs = '''
dV/dt = -V/(10*ms) : volt
'''
G = NeuronGroup(10, eqs)
G.V = randn(10)*10*mV
M = StateMonitor(G, 'V', record=True)

V = copy(G.V)

blG = bl.NeuronGroup(G._S)
blGsu = bl.LinearStateUpdater(G._state_updater.A, G._state_updater._C.flatten())

v = []
for _ in xrange(1000):
    blGsu.__call__(blG)
    v.append(blG.get_S_flat(10))

subplot(121)
plot(array(v))
title('C++')
subplot(122)
G.V = V
run(100*ms)
for i in range(10):
    plot(M[i])
title('Brian')
show()