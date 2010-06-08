from brian import *
from brian.library.ionic_currents import *
from brian.experimental.codegen import *
import time
from scipy import weave

N = 1000
record_and_plot = N == 1

El = 10.6 * mV
EK = -12 * mV
ENa = 120 * mV
eqs = MembraneEquation(1 * uF) + leak_current(.3 * msiemens, El)
eqs += K_current_HH(36 * msiemens, EK) + Na_current_HH(120 * msiemens, ENa)
eqs += Current('I:amp')
eqs.prepare()

print eqs
print '.............................'
pycode = PythonCodeGenerator().generate(eqs, exp_euler_scheme)
print pycode
print '.............................'
ccode = CCodeGenerator().generate(eqs, exp_euler_scheme)
print ccode

neuron = NeuronGroup(N, eqs, implicit=True, freeze=True)

if record_and_plot:
    trace = StateMonitor(neuron, 'vm', record=True)

neuron.I = 10 * uA

_S_python = array(neuron._S)
_S = array(neuron._S)

ns = {'_S':_S_python, 'exp':exp, 'dt':defaultclock._dt}
pycode_comp = compile(pycode, '', 'exec')

start = time.time()
run(100 * ms)
print 'N:', N
print 'Brian:', time.time() - start

start = time.time()
hand_trace_python = []
for T in xrange(int(100 * ms / defaultclock.dt)):
    exec pycode_comp in ns
    if record_and_plot:
        hand_trace_python.append(copy(_S_python[0]))
print 'Codegen Python:', time.time() - start

start = time.time()
hand_trace_c = []
for T in xrange(int(100 * ms / defaultclock.dt)):
    dt = defaultclock._dt
    t = T * defaultclock._dt
    num_neurons = len(neuron)
    weave.inline(ccode, ['_S', 'num_neurons', 'dt', 't'],
                 compiler='gcc',
                 #type_converters=weave.converters.blitz,
                 extra_compile_args=['-O3', '-march=native', '-ffast-math'])#O2 seems to be faster than O3 here
    if record_and_plot:
        hand_trace_c.append(copy(_S[0]))
print 'Codegen C:', time.time() - start

if record_and_plot:
    plot(trace[0])
    plot(array(hand_trace_python))
    plot(array(hand_trace_c))
    show()
