from brian import *
from brian.utils.circular import *
import brianlib as bl
debugmode=False

def f(s):
    if s.__class__ is SpikeContainer:
        t='SpikeContainer\n'
        t+='  S: CircularVector(cursor='+str(s.S.cursor)+', X=['+' '.join(map(str, s.S.X))+'])\n'
        t+='  ind: CircularVector(cursor='+str(s.ind.cursor)+', X=['+' '.join(map(str, s.ind.X))+']))'
        return t
    else:
        return 'brianlib.'+repr(s)

sc=SpikeContainer(12, 6)
blsc=bl.SpikeContainer(12, 6)
for i in range(20):
    for s in [sc, blsc]:
        x=array([1, 2, 3, 4])+10*i
        s.push(x)
        #s.push(bl.SpikeList(x))
        print f(s)
        print s.get_spikes(0, 0, 200)
        print s.lastspikes()


if False:
    v1=CircularVector(6, dtype=int)
    v2=bl.CircularVector(6)
    for v in [v1, v2]:
        v[0:4]=array([1, 2, 3, 4])
        if debugmode:
            print 'First push OK'
            print v
        print v[0:4], v[4:6]
        v.advance(4)
        if debugmode:
            print 'advance 4'
            print v
        print v[0:4], v[4:6]
        v[0:4]=array([5, 6, 7, 8])
        if debugmode:
            print 'Second push OK'
            print v
        print v[0:4], v[4:6]
        v.advance(4)
        if debugmode:
            print 'advance 4'
            print v
        print v[0:4], v[4:6]
