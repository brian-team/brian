from brian import *
set_global_preferences(useweave=False, usecodegen=False)
tau=1*ms
sigmax=1*mV
sigmay=2*mV

# These equations don't work
#eqs="""
#dx/dt=-x/tau+sigmax*u/tau**.5 : volt
#dy/dt=-y/tau+sigmay*u/tau**.5 : volt
#u=xi : second**(-.5)
#"""
# Workaround
eqs="""
v = x+sigmax*u : volt
w = y+sigmay*u : volt
dx/dt=-x/tau : volt
dy/dt=-y/tau : volt
du/dt=xi/tau**.5 : 1
"""
neuron=NeuronGroup(1,model=eqs)
#M=MultiStateMonitor(neuron, record=True)
M=MultiStateMonitor(neuron, ['v', 'w'], record=True)
run(10*ms)
M.plot()
show()
