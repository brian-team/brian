if __name__ == '__main__':

    from numpy import dot, eye
    from numpy.linalg import inv
    from distfun import *
    
    A = 2*eye(3,3)
    B = 3*eye(3,3)
    
    distinv = DistributedFunction(inv, max_cpu=2)
    invA, invB = distinv([A,B])
    
    print invA
    print invB

