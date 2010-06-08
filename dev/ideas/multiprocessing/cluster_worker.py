from numpy import *
from multiprocessing.connection import Listener
from array import array

def f(x, a, n):
    s = 0
    for _ in xrange(n):
        s += sum(x) * a
    return s

address = ('localhost', 2718)     # family is deduced to be 'AF_INET'
listener = Listener(address, authkey='secret password')
print 'Listener created'

conn = listener.accept()
print 'connection accepted from', listener.last_accepted

x = conn.recv()

while True:
    item = conn.recv()
    if item is None:
        break
    a, n = item
    conn.send(f(x, a, n))

listener.close()
conn.close()
