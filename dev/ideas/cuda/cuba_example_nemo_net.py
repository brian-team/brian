use_nemo = True
do_callgraph = False
plot_output = True

from brian import *
if use_nemo:
    from brian.experimental.cuda.briantonemo import *
import time

eqs = '''
dv/dt = (ge-gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P = NeuronGroup(4000, model=eqs, threshold=-50*mV, reset=-60*mV)
P.v = -60 * mV + 10 * mV * rand(len(P))
Pe = P.subgroup(3200)
Pi = P.subgroup(800)

Ce = Connection(Pe, P, 'ge', weight=1.62*mV, sparseness=0.02,
                delay=(0*ms, 0*ms),
                )
Ci = Connection(Pi, P, 'gi', weight=9*mV, sparseness=0.02,
                delay=(0*ms, 0*ms),
                )

M = SpikeMonitor(P)

net = MagicNetwork()

start = time.time()
net.prepare()
net.run(1*ms)
print 'Preparation time:', time.time()-start

if do_callgraph:
    import pycallgraph
    def ff(call_stack, module_name, class_name, func_name, full_name):
        if not 'brian' in module_name: return False
        return True
    pycallgraph.start_trace(filter_func=ff)

start = time.time()
net.run(1 * second)
print 'Run time:', time.time()-start

if do_callgraph:
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('callgraph.png')

if plot_output:
    raster_plot(M)
    show()
