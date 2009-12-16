from numpy import *
from multiprocessing.connection import Client

N = 10000
numprocesses = 6
complexity = 10

address = ('localhost', 6000)
conn = Client(address, authkey='secret password')
print 'Connection acquired'

x = ones(N)

conn.send(x)
print 'Shared data sent'

results = []
for a in xrange(numprocesses):
    conn.send((a, complexity))
    results.append(conn.recv())

print results

conn.send(None)

conn.close()