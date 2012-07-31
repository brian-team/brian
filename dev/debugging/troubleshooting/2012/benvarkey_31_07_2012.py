import brian_no_units
import brian
from pylab import *

eq1=brian.Equations('g=gmax*c0 :volt')
#eq1+=brian.Equations('g_ch : 1') #uncomment this to make it work
eq1+=brian.Equations('dc0/dt=-1 :volt')
eq1+=brian.Equations('gmax :1')
g1=brian.NeuronGroup(1, model=eq1)

eq2=brian.Equations('dv/dt=-v+gch*(erev-v): volt')
eq2+=brian.Equations('erev :1')
eq2+=brian.Equations('gch :1')
g2=brian.NeuronGroup(1, model=eq2)

#g1.g_ch = brian.linked_var(g1, 'g') #uncomment this to make it work

g2.gch = brian.linked_var(g1, 'g') #comment this to make it work
#g2.gch = brian.linked_var(g1, 'g_ch') #uncomment this to make it work

s1=brian.StateMonitor(g1, 'g', record=0)
s2=brian.StateMonitor(g1, 'c0', record=0)

s3=brian.StateMonitor(g2, 'v', record=0)
s4=brian.StateMonitor(g2, 'gch', record=0)

s5=brian.StateMonitor(g1, 'gmax', record=0)

#initialize
g1.g=0.0 #commenting this ALONE will make it work
#g1.g_ch=0.0 #uncomment this to make it work
g1.gmax=2.0
g2.erev=2.0

n=brian.Network()
n.add(g1)
n.add(g2)
n.add((s1, s2, s3, s4, s5))
n.run(1)

plot(s4.times, s4[0])
show()