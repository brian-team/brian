from brian import *
from brian.compartments import *
from brian.library.synapses import *

Ces=200*pF
Ced=200*pF
taue=5*ms
taui=10*ms
Ee=0*mV
Ei=-80*mV
El=-70*mV
gl=20*nS
gld=20*nS
Ra=1*Mohm

soma = MembraneEquation(C=Ces) +  Current('I=gl*(El-vm):amp')
soma += alpha_conductance(input='ge' ,E=Ee,tau=taue)
soma += alpha_conductance(input='gi' ,E=Ei,tau=taui)

dend = MembraneEquation(C=Ced) +  Current('I=gld*(El-vm):amp')
dend += alpha_conductance(input='ged' ,E=Ee,tau=taue,conductance_name='gee')
dend += alpha_conductance(input='gid' ,E=Ei,tau=taui)

neuron_eqs=Compartments({'soma':soma,'dendrite':dend})
neuron_eqs.connect('soma','dendrite',Ra)

neuron=NeuronGroup(2,model=neuron_eqs)

connexion_EE=Connection(neuron,neuron,'gee_dendrite')