"""
Memory consumption of the Synapses vs. Connection object

All-to-all connections
N=5000 -> 25e6 synapses

1054

Synapses:
1 variable:           611 MB -> 24B/syn
3 variables:          985 MB
3 vars + pre/post:    1037 MB -> 41B/syn

Connection (sparse but no delay):
1151 MB -> 46B/syn
DelayConnection (1 ms max delay)
1164 MB
DelayConnection + STDP (1 ms max delay)
1155 MB
"""
from brian import *
from time import time
import sys, gc
from brian.experimental.synapses import *
#from guppy import hpy
#h = hpy()

N=3000

#print h.heap()

def countbytes(obj, name='', found=None, printsizes=True):
    if found is None:
        found = set()
    if id(obj) in found:
        return 0
    found.add(id(obj))
    totalbytes = sys.getsizeof(obj)
    if hasattr(obj, 'nbytes'):
        if hasattr(obj, 'base') and isinstance(obj.base, ndarray):
            return countbytes(obj.base, name=name+'.base', found=found)
        nb = obj.nbytes
        if isinstance(nb, int):
            totalbytes += nb
    if hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.iteritems():
            if not isinstance(v, NeuronGroup):
                nb = countbytes(v, name=name+'.'+k, found=found)
                totalbytes += nb
    if isinstance(obj, list):
        if len(obj):
            if isinstance(obj[0], (int, float)):
                # assume it's homogeneous
                first = countbytes(obj[0], name=name+'[0]', found=found)
                sz = len(obj)*first
                totalbytes += sz
            else:                
                for i, v in enumerate(obj):
                    totalbytes += countbytes(v, name=name+'['+str(i)+']',
                                             found=found, printsizes=False)
    if isinstance(obj, ndarray) and obj.dtype==object:
        for v in obj:
            totalbytes += countbytes(v, name=name+'[something]', found=found)
    if printsizes and totalbytes>1024*1024:
        print 'Size of', name, '=', totalbytes/1024**2, 'MB'
    return totalbytes

#1740 -> 574
P=NeuronGroup(N,model="v:1")
k=1
if k==0:
    S=Synapses(P,model='''w:1
                          Apre:1
                          Apost:1''',pre='v+=w',post='w+=Apre')
    S[:,:]=True
    nb = countbytes(S, 'S')
    print 'S uses', nb/1024**2, 'MB (%d)'%nb
elif k==1:
    C=Connection(P,P,'v',delay=True,max_delay=1*ms)
    C.connect_full(P,P,weight=1.,delay=0.1*ms)
    print 'Before compression'
    print
    nb = countbytes(C, 'C')
    print
    print 'Before compression C uses', nb/1024**2, 'MB (%d)'%nb
    print C.W.data[0].__class__, C.W.data[0][0].__class__
    print C.W.rows[0].__class__, C.W.rows[0][0].__class__
    gc.collect()
    _=raw_input("Press enter")
    C.compress()
#    print C.W.__class__
    print
    print 'After compression'
    print
    nb = countbytes(C, 'C')
    print
    print 'After compression C uses', nb/1024**2, 'MB (%d)'%nb
    print 'Expected size of C', 2*16*N*N/1024**2, 'MB'
elif k==2:
    C=Connection(P,P,'v',delay=True,max_delay=1*ms)
    C.connect_full(P,P,weight=1.)
    S=ExponentialSTDP(C, 10*ms, 10*ms, 1., 1.,wmax=1.)

gc.collect()
_=raw_input("Press enter")
#print h.heap()
