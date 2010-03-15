from numpy import *
from distributedfunction import *

def square(x):
    """
    Must be defined in the global namespace.
    """
    return x**2

if __name__ == '__main__':
    dsquare = distribute(square)
    x = [1,2,3,4]
    y = dsquare(x)
    print y
    
    