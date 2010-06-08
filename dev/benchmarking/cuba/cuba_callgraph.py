from brian import *
set_global_preferences(useweave=True)
import cuba_runopts
cuba_runopts.duration = 100 * ms
import pycallgraph
from cuba import *

cg_func = 'Connection.do_propagate'

def ff(pat):
    def f(call_stack, module_name, class_name, func_name, full_name):
        if not 'brian' in module_name: return False
        for n in call_stack + [full_name]:
            if pat in n:
                return True
        return False
    return f

def cuba(P, Pe, Pi, Ce, Ci, M, net):
    net.run(duration)

c = cubanetwork(4000)
pycallgraph.start_trace(filter_func=ff(cg_func))
cuba(*c)
pycallgraph.stop_trace()
pycallgraph.make_dot_graph('callgraphs/cuba-callgraph-' + cg_func + '.png')
