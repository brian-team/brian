# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
"""
Brian bottom-up network builder

STILL LOTS OF WORK TO DO ON THIS, SAVE FOR FUTURE RELEASE?
"""

from connection import *
from stateupdater import *
from neuronmodel import *
from neurongroup import *
from network import *
from units import *
from collections import defaultdict
from globalprefs import *
from operator import isSequenceType

class NeuronModel(Neuron):
    def __mul__(self,other):
        if isinstance(other,int):
            return Group(N=other,model=self)
        else:
            raise ValueError
    def __rmul__(self,other):
        if isinstance(other,int):
            return Group(N=other,model=self)
        else:
            raise ValueError

class Group(object):
    def __init__(self,N,model,*args,**kwds):
        self.N = N
        self.model = model
        self.args = args
        self.kwds = kwds
        self.connects = set()
    def __call__(self):
        return self.neurongroup

class MultiGroup(object):
    def __init__(self,*groups):
        self.groups = groups

class Connect(object):
    def __init__(self,source,target,state=0,function=None,*args,**kwds):
        self.source = source
        self.target = target
        self.state = state
        self.function = function
        self.args = args
        self.kwds = kwds
        source.connects.add(self)
        target.connects.add(self)
    def signature(self):
        return (id(self.source.model),id(self.target.model),self.state)
    def __call__(self):
        return self.connection

class ConnectFull(Connect):
    def __init__(self,source,target,state=0,weight=1.,*args,**kwds):
        Connect.__init__(self,source,target,state=state,function='connect_full',weight=weight,*args,**kwds)

class ConnectRandom(Connect):
    def __init__(self,source,target,state=0,p=0.1,weight=1.,*args,**kwds):
        Connect.__init__(self,source,target,state=state,function='connect_random',weight=weight,p=p,*args,**kwds)

class Net(Network):
    def __add(self,o):
        if isinstance(o,Group):
            self.netgroups.add(o)
        elif isinstance(o,Connect):
            self.netconnects.add(o)
        elif isSequenceType(o):
            map(self.__add,o)
        else: # use the Network add, it's some other type of object that Net doesn't know about
            self.add(o)
    def __init__(self,*objs,**kwds):
        Network.__init__(self)
        self.netgroups = set()
        self.netconnects = set()
        for o in objs:
            self.__add(o)
        groups = self.netgroups
        connects = self.netconnects
        for g in groups:
            connects |= g.connects
        # these correspond to the NeuronGroup objects
        groupsbymodel = defaultdict(list)
        for g in groups:
            groupsbymodel[id(g.model)].append(g)
        neurongroup = {}
        for k, G in groupsbymodel.iteritems():
            N = sum(g.N for g in G)
            neurongroup[k] = NeuronGroup(N=N,model=G[0].model)
            for g in G:
                g.neurongroup = neurongroup[k].subgroup(g.N)
                g.net = self
            print 'neurongroup',k,neurongroup[k]
        # these correspond to the Connection objects
        connectsbysignature = defaultdict(list)
        for c in connects:
            connectsbysignature[c.signature()].append(c)
        connection = {}
        for k,C in connectsbysignature.iteritems():
            connection[k] = Connection(neurongroup[k[0]],neurongroup[k[1]],state=k[2])
            print "connection", k, connection[k]
            # these correspond to the Connection function calls
            for c in C:
                c.connection = connection[k]
                c.net = self
                if c.function is not None:
                    getattr(connection[k],c.function)(c.source(),c.target(),*c.args,**c.kwds)
            print connection[k].W
        # now let Network take over...
        for o in neurongroup.values()+connection.values():
            self.add(o)

if __name__=='__main__':
    set_global_preferences(useweave=False)
    model1 = NeuronModel(model=LazyModel(),reset=0*mvolt,threshold=50*mvolt,init=0*mvolt)
    model2 = NeuronModel(model=LazyModel(numstatevariables=2),reset=0*mvolt,threshold=20*mvolt,init=(0*mvolt,0*mvolt))
    g = [ 10*model1, 20*model2, 30*model1 ]
    ConnectFull(g[0],g[1],0,weight=1*mvolt),
    ConnectFull(g[0],g[1],1,weight=3*mvolt),
    ConnectRandom(g[2],g[1],0,p=0.1,weight=2*mvolt)
    net = Net(g)
    net.run(10*msecond)