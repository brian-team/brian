from brian import *
import pickle

eqs = Equations('''
dV/dt=-V/(10*ms):1
''')

eqs.prepare()

#print eqs._namespace

s = pickle.dumps(eqs)
eqs2 = pickle.loads(s)

print eqs2