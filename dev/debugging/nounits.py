from brian import *

eqs='''
dv/dt=-v*invtaudiff : 1
invtaudiff = 1/(tau1-tau2) : 1/second
tau1 : ms
tau2 : ms
'''

G=NeuronGroup(N=1, model=eqs, threshold=1, reset=0, unit_checking=False)
