if __name__ == '__main__':

    from numpy import dot
    from numpy.random import rand
    from numpy.linalg import inv
    from distfun import *
    
    A = rand(4,4)
    B = rand(4,4)
    
    distinv = DistributedFunction(inv, max_cpu=2)
    invA, invB = distinv([A,B])
    
    print dot(A,invA)
    print dot(B,invB)

