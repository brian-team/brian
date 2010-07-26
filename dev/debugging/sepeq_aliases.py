from brian import *
from brian.utils.separate_equations import *

eqs = Equations('''
dx/dt = -x/second : 1
dy/dt = -y/second : 1
z     = x-y       : 1
du/dt = -u/second : 1
dv/dt = -v/second : 1
w     = u-v       : 1
''')

print separate_equations(eqs)

eqs = Equations('''
dx/dt = -x/second : 1
y = x*x : 1
z = x
du/dt = -u/second : 1
#v = z+u : 1
''')

print separate_equations(eqs)
