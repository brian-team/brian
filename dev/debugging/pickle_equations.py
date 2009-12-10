from brian import *
import pickle

eqs = Equations('''
dV/dt=-V/(10*ms):1
''')

s = pickle.dumps(eqs)
