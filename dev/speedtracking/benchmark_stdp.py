from datetime import datetime

from vbench.benchmark import Benchmark

common_setup = """
from brian import *
log_level_error() # do not show warnings    
"""

python_only_setup = """
set_global_preferences(useweave=False, usecodegen=False, usecodegenweave=False)
"""

setup= """
N = 1000
taum = 10 * ms
tau_pre = 20 * ms
tau_post = tau_pre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
El = -74 * mV
taue = 5 * ms
gmax = 0.01
F = 15 * Hz
dA_pre = .01
dA_post = -dA_pre * tau_pre / tau_post * 2.5

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, model=Equations(eqs_neurons), threshold=vt,
                      reset=vr)
synapses = Connection(input, neurons, 'ge', weight=rand(len(input), len(neurons)) * gmax,
                    structure='dense')
neurons.v = vr

stdp = ExponentialSTDP(synapses, tau_pre, tau_post, dA_pre, dA_post, wmax=gmax, update='mixed')
run(defaultclock.dt)                     
"""

statement = '''
run(5 * second)
'''

bench_stdp = Benchmark(statement, common_setup + python_only_setup + setup,
                       name='Exponential STDP (Python only)')
