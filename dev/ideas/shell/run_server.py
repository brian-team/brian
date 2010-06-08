from brian import *
from multiprocessing.connection import Listener, Client
import select

server = ('localhost', 2719)
server_authkey = 'brian'

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
P = NeuronGroup(4000, model=eqs,
              threshold= -50 * mV, reset= -60 * mV)
P.v = -60 * mV + 10 * mV * rand(len(P))
Pe = P.subgroup(3200)
Pi = P.subgroup(800)
Ce = Connection(Pe, P, 'ge', weight=1.62 * mV, sparseness=0.02)
Ci = Connection(Pi, P, 'gi', weight= -9 * mV, sparseness=0.02)
M = SpikeMonitor(P)
@network_operation(clock=EventClock(dt=1 * second))
def clearspikes():
    M.reinit()
    print 'Simulated', defaultclock.t

listener = Listener(server, authkey=server_authkey)
#conn = listener.accept()
conn = None

global_ns = globals()
local_ns = locals()

@network_operation
def server_check():
    global conn
    if conn is None:
        socket = listener._listener._socket
        sel, _, _ = select.select([socket], [], [], 0)
        if len(sel):
            conn = listener.accept()
    if conn is None:
        return
    if not conn.poll():
        return
    job = conn.recv()
    jobtype, jobargs = job
    if jobtype == 'exec':
        exec jobargs in global_ns, local_ns
        result = None
    elif jobtype == 'eval':
        result = eval(jobargs, global_ns, local_ns)
    conn.send(result)

run(1e10 * second)

print 'Finished!'
