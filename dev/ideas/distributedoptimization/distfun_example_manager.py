from numpy import *
from distfun import *

def square(x):
    """
    Must be defined in the global namespace.
    """
    return x**2

if __name__ == '__main__':
    dsquare = DistributedFunction(square, machines = ['localhost'], named_pipe=None)
    x = arange(1,16)
    y = dsquare(x)
    print y
