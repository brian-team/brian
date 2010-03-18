"""
Distributed function quick example 2
************************************

This example shows how to distribute a "matrix-to-vector"-like function 
over several workers. If you're using Python/Numpy for intensive numerical 
computations, you will probably want to use this library like this.
"""

"""
In this example, we want to distribute a function ``f`` that accepts any
D-long vector and returns a number. Very often, it is possible to vectorize
this function using Numpy matrices operations, in such a way that the vectorized function 
``fun`` takes a DxN matrix ``x`` as an argument, and returns a N-long vector. 
We then have ``fun(x)[i] == f(x[:,i])``. This is a "matrix-to-vector"-like function.

In this case, it becomes easy to distribute this function over multiple 
CPUs/machines. The matrix argument ``x`` is divided into D*K submatrices of approximate
equal size. Each worker calls ``fun`` with the corresponding submatrix,
and the manager concatenates the results of the workers in a transparent way.
The parallelization is then totally transparent to the user.

In the following example, the function ``f`` computes the sum of the components
of a D-long vector. The Numpy function ``sum`` can do this in a vectorized way :
if ``x`` is a DxN matrix, ``sum(x, axis=0)`` is an N-long vector containg 
the sum of the components of each column of ``x``.
"""

"""
We define the function to distribute. It must accept a DxN matrix and return a N-long
vector, where ``fun(x)[i] == f(x[:,i])`` for a given function ``f``.
"""
from numpy import sum
def fun(x):
    return sum(x, axis=0)

if __name__ == '__main__':
    from numpy import ones
    
    """
    We import the library.
    """
    from distfun import *
    
    """
    This is the most important line of this example. It defines the
    distributed version of the function ``fun ``, with no more than four CPUs.
    """
    dfun = DistributedFunction(fun, max_cpu=4)
    
    """
    This is the matrix that we want to pass to ``fun``. The call to
    ``y=fun(x)`` is a perfectly valid Python statement and returns the right result,
    but runs over a single CPU, even if several CPUs are available in the machine.
    """
    x = ones((5,8))
    
    """
    The call to ``y=dfun(x)`` returns exactly the same result as ``y=fun(x)``, but
    distribute the execution of the function over several workers. By default,
    the library creates one worker per available CPU in the machine.
    
    Here, if there are two CPUs in the system, each one will execute ``fun(subx)`` 
    with ``subx`` being the left half of ``x`` for CPU 1, and the right half for 
    CPU 2.
    """
    y = dfun(x)
    print y
    
    """
    That is probably the most efficient way of using this library. If you're 
    already using Numpy for numerical computations, it should be straighforward
    to define a "matrix-to-vector"-like function that performs your computations.
    You basically have nothing to do to execute it over several CPUs.
    It is almost as easy to distribute it over several machines connected in a network,
    see the network example. 
    """