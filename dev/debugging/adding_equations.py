from brian import *

bof = 3.0
tau = 10 * ms

eq_base = Equations('''
w = V*bof : 1
''')

eq_extra_1 = Equations('''
dV/dt = (w-V)/tau : 1
''')

eq_extra_2 = Equations('''
dV/dt = (2*w-V)/tau : 1
''')

eq1 = eq_extra_1 + eq_base
eq2 = eq_extra_2 + eq_base

print eq1._namespace['w'].keys()

eq1.prepare()

print eq1._namespace['w'].keys()

eq2.prepare()

#exit()

G1 = NeuronGroup(1, eq1)
G2 = NeuronGroup(1, eq2)

run(1 * ms)

print G1.V
print G1.w
print G2.V
print G2.w
