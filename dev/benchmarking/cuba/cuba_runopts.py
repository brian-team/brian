from brian import *

# long run vals
Nvals = [1000, 2000, 4000, 8000, 16000, 32000]
N_varycon = 1000
N_varywe = 4000
Nsynvals = [80., 200., 500., 1000.]
wevals = [1.62*mV, 3*mV, 6*mV, 9*mV]
duration = 2.5*second
repeats = 10
best = 7
cuba_opts = {'connections':True}
# debug run vals
#Nvals = [1000, 2000]
#duration = 2.5*second
#repeats = 3
#best = 2
