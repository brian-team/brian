from brian import *
import pycallgraph

dorun = True

def ff(call_stack, module_name, class_name, func_name, full_name):
    if func_name=='cuba': return True
    if not 'brian' in module_name: return False
    if func_name=='__len__': return False
    if 'units' in module_name: return False
    return True

def ff_nomagicstateupdater(call_stack, module_name, class_name, func_name, full_name):
    if func_name=='cuba': return True
    if not 'brian' in module_name: return False
    if func_name=='__len__': return False
    if 'units' in module_name: return False
    if func_name=='magic_state_updater' or func_name=='prepare':
        return True
    if 'magic' in module_name: return False
    for n in call_stack+[full_name]:
        if 'magic_state_updater' in n:
            return False
        if 'prepare' in n:
            return False
    return True

def ff_magicstateupdater(call_stack, module_name, class_name, func_name, full_name):
    if func_name=='cuba': return True
    if not 'brian' in module_name: return False
    if func_name=='__len__': return False
    if 'units' in module_name: return False
    if func_name=='prepare':
        return True
    for n in call_stack+[full_name]:
        if 'prepare' in n:
            return False
    for n in call_stack+[full_name]:
        if 'magic_state_updater' in n:
            return True
    return False

def ff_prepare(call_stack, module_name, class_name, func_name, full_name):
    if func_name=='cuba': return True
    if not 'brian' in module_name: return False
    if func_name=='__len__': return False
    if 'units' in module_name: return False
    for n in call_stack+[full_name]:
        if 'prepare' in n:
            return True
    return False

def ff_noprepare(call_stack, module_name, class_name, func_name, full_name):
    if func_name=='cuba': return True
    if not 'brian' in module_name: return False
    if func_name=='__len__': return False
    if 'units' in module_name: return False
    if func_name=='prepare': return True
    for n in call_stack+[full_name]:
        if 'prepare' in n:
            return False
    return True

def cuba():
    N = 4000
    Ne = int(N*0.8) 
    Ni = N-Ne
    p = 80./N
    
    eqs='''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''

    pycallgraph.start_trace(filter_func=ff_nomagicstateupdater)
    P=NeuronGroup(N, eqs,
                  threshold=-50*mV, reset=-60*mV)
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('callgraphs/cuba-bigcallgraph-NeuronGroup.__init__.no_magic_state_updater.png')    
    
    P.v = -60*mV+10*mV*rand(len(P))
    Pe = P.subgroup(Ne)
    Pi = P.subgroup(Ni)
    
    pycallgraph.start_trace(filter_func=ff)
    Ce = Connection(Pe,P,'ge',weight=1.62*mV,sparseness=p)
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('callgraphs/cuba-bigcallgraph-Connection.__init__.png')    

    Ci = Connection(Pi,P,'gi',weight=-9*mV,sparseness=p)
    
    pycallgraph.start_trace(filter_func=ff)
    M = SpikeMonitor(P)
    trace = StateMonitor(P,'v',record=0)
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('callgraphs/cuba-bigcallgraph-Monitors.__init__.png')    
    
    pycallgraph.start_trace(filter_func=ff_prepare)
    run(.1*ms)
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('callgraphs/cuba-bigcallgraph-run-prepare.png')

cuba()

def cuba():
    N = 4000
    Ne = int(N*0.8) 
    Ni = N-Ne
    p = 80./N
    
    eqs='''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''

    pycallgraph.start_trace(filter_func=ff_magicstateupdater)
    P=NeuronGroup(N, eqs,
                  threshold=-50*mV, reset=-60*mV)
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('callgraphs/cuba-bigcallgraph-NeuronGroup.__init__.magic_state_updater.png')    

    P.v = -60*mV+10*mV*rand(len(P))
    Pe = P.subgroup(Ne)
    Pi = P.subgroup(Ni)
    
    Ce = Connection(Pe,P,'ge',weight=1.62*mV,sparseness=p)
    Ci = Connection(Pi,P,'gi',weight=-9*mV,sparseness=p)
    
    M = SpikeMonitor(P)
    trace = StateMonitor(P,'v',record=0)
    
    pycallgraph.start_trace(filter_func=ff_noprepare)
    net = MagicNetwork()
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('callgraphs/cuba-bigcallgraph-run-noprepare-netinit.png')    
    
    pycallgraph.start_trace(filter_func=ff_noprepare)
    net.run(.1*second)
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('callgraphs/cuba-bigcallgraph-run-noprepare-nonetinit.png')    

cuba()

def cuba():
    eqs='''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''
    eqs = Equations(eqs)
    pycallgraph.start_trace(filter_func=ff)
    eqs.prepare()
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('callgraphs/cuba-bigcallgraph-Equations.prepare.png')    
    
cuba()