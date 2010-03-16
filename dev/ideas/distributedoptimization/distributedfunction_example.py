def square(x):
    """
    The function you want to execute in parallel. It can accept only one argument.
    The argument can be :
    - an array : each core executes the function with a different view on the array
    - a list : each core executes a sublist of the original list
    - any object : each core executes the function with a different object
    
    Must be defined in the global namespace.
    
    Here, the 'square' function accepts a number or an array as an argument,
    and puts to square all its elements (element-wise operation).
    """
    return x**2

"""
IMPORTANT NOTE FOR WINDOWS USERS:
The code *must* be placed after "if __name__ == '__main__':"
to avoid infinite loops which would make the computer crash.
"""
if __name__ == '__main__':
    import distributedfunction as df
    import numpy as np
    
    """
    A distributed function 'dsquare' is created from the function 'square'.
    """
    dsquare = df.distribute(square)
    
    """
    The list of arguments we want to evaluate 'square' with. It can be
    a vector of any size here.
    """
    x = np.arange(12)
    
    """
    We evaluate the function over our parameters in parallel in a transparent way.
    By default, the library equally divides 'x' into as many views as there are CPUs on the system.
    Each CPU computes 'square' with a chunck of 'x', and the results are then combined into 'y'.
    Here, with a dual-core for example :
        - core 1 computes square([0,1,2,3,4,5])
        - core 2 computes square([6,7,8,9,10,11)
    'y' is then the concatenation of the results.
    """
    y = dsquare(x)
    print y
    
    