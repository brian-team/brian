'''
A client script to be run for cluster simulations.
'''
from brian import *
import pypar

__all__=['run_client']

class ClientNetwork(Network):
    '''
    Network class for running a simulation over a cluster, client side.
    '''
    pass

def run_client():
    '''
    Runs
    '''
    # Identification
    myid =    pypar.rank() # id of this process
    nproc = pypar.size() # number of processors

    print "I am client",myid
    pypar.finalize()

if __name__=='__main__':
    run_client()
