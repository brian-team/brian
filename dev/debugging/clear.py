from brian import *
from brian.network import clear

G = NeuronGroup(1, 'V:1')

clear(True)

print MagicNetwork().groups
print G._S