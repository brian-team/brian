from numpy import *
from distributedfunction import *

def square(x):
    """
    Must be defined in the global namespace.
    """
    return x**2

if __name__ == '__main__':
    dsquare = DistributedFunction(square, machines = [], named_pipe=True)
    x = arange(1,6)
    y = dsquare(x)
    print y
