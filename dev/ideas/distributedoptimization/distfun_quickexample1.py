"""
Distributed function quick example 1
************************************

This example shows how to distribute a "one parameter-one result"-like function
over several workers.
"""

"""
For Windows users, it is required that any code using this library is placed after
this line, otherwise the system will crash!
"""
if __name__ == '__main__':
    
    from numpy import eye
    from numpy.linalg import inv
    
    """
    We import the library to have access to the ``DistributedFunction`` class,
    which is the main class used to distribute a function over multiple CPUs
    or machines.
    """
    from distfun import *
    
    """
    This is the most important instruction of this example. It defines a 
    distributed version of the Numpy ``inv`` function which inverses any square matrix.
    
    The first argument of ``DistributedFunction`` is the name of the function
    that is being distributed. This function must accept a single argument and return a
    single object. The optional argument ``max_cpu=n`` allows to limit the number
    of CPUs that are going to be used on the current machine by the distributed function. 
    Of course, this has no effect if there are less than n CPUs available in the machine.
    """
    distinv = DistributedFunction(inv, max_cpu=2)
    
    """
    We define the two matrices that are going to be inversed in parallel.
    """
    A = 2*eye(3)
    B = 4*eye(3)
    
    """
    ``distinv`` is the distributed version of ``inv`` : it is called by passing
    a list of arguments. The original function ``inv`` is automatically called 
    once per argument in the list in a distributed fashion. The distributed function
    returns the list of the results of each call.
    
    The list can be of any size. If there are more arguments
    than workers, then each worker will process several arguments in series.
    Here, if there are two available CPUs in the system, the first CPU inverses
    A, the second inverses B. ``invA`` and ``invB`` contain the inverses of A and B.
    """
    invA, invB = distinv([A,B])
    
    print invA
    print invB
    
    """
    That is the simplest way of distribute a function. It is used typically for
    functions that perform complex and costly operations on a single object.
    Distributing this function allows to perform the same operations on different
    objects in parallel.
    
    Go to quick example 2 to see how you can distribute Python functions that are
    already vectorized thanks to Numpy matrices operations.
    """