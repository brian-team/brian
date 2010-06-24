from brian import *
nbr_neurons=2

gl=2*nS*linspace(0, 1, nbr_neurons)
El=-65*mV*ones(nbr_neurons)
#C=12*pF*ones(nbr_neurons) ## that was changed from previous example
C=12*pF
eqs_leak="""
ileak = gl*(El-v) : amp
"""
eqs="""
dv/dt=(ileak)/C : volt
"""
eqs+=eqs_leak
neuron=NeuronGroup(nbr_neurons,model=eqs, compile=True)#method='Euler')
M = StateMonitor(neuron, 'v', record=True)
run(100*ms,report='text')
M.plot()
show()
