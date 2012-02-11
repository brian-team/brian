"""
Test the consistency of synapses_post and postsynaptic
"""
from brian import *
from brian.experimental.synapses import *

N=100
P=NeuronGroup(N,'v:1')
S=Synapses(P,P,model="w:1",pre="v+=w")
S[:,:]=0.1

for i in range(N):
    if (S.presynaptic[S.synapses_pre[i][:]]!=i).any():
        print "pre: inconsistency!"

for i in range(N):
    if (S.postsynaptic[S.synapses_post[i][:]]!=i).any():
        print S.postsynaptic[S.synapses_post[i][:]]

#postsynaptic=randint(5,size=20)
#synapses_post=invert_array(postsynaptic)
#print postsynaptic,synapses_post
#print
#
#for i,synapses in synapses_post.iteritems():
#    if (postsynaptic[synapses]!=i).any():
#        print i,postsynaptic[synapses]
