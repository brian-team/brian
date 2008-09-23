'''
Pickling example, see also pickle_savenet.py
'''

from brian import *
import pickle

inputfile = open('data.pkl','rb')
obj = pickle.load(inputfile)
inputfile.close()

clk, eqs, P, Pe, Pi, Ce, Ci, M, net = obj

net.run(100*ms)
raster_plot(M)
show()