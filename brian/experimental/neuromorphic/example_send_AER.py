# -*- coding:utf-8 -*-
"""
"""
from brian import *
from brian.experimental.neuromorphic import *

N=128
input=PoissonGroup(N*N,rates=10*Hz)

M=SpikeMonitor(input)

run(1*second)

save_AER(M,'example.aedat')
