from brian import *

G = PoissonGroup(1000, 50*Hz)
T = 10*ms
M = ISIHistogramMonitor(G, bins=[0*ms,T,2*T,3*T,4*T])
run(1*second)
print M.count
hist_plot(M)
show()