from brian import *

tau=1*ms
eqs=Equations('dx/dt=-x/tau : 1',x='y')
eqs.prepare()
