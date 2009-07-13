from brian import *
from brian.experimental.cuda.gpucodegen import *

use_gpu = True

tau = 200*ms
N = 3

eqs = Equations('''
dV/dt = (-V+I)/tau : 1
I : 1
''')

if use_gpu:
    G = GPUNeuronGroup(N, eqs, gpu_to_cpu_vars=['V'], cpu_to_gpu_vars=['I'])
else:
    G = NeuronGroup(N, eqs)

@network_operation
def f():
    G.I = float(defaultclock.t)*arange(len(G))

M = StateMonitor(G, 'V', record=True)

G.V = 0
G.I = 0
if use_gpu:
    G._S.sync_to_gpu()

run(1*second)

M.plot()
show()