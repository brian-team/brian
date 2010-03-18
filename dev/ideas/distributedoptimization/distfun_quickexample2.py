from numpy import sum

def fun(x):
    return sum(x, axis=0)

if __name__ == '__main__':
    from numpy import ones
    from distfun import *
    dfun = DistributedFunction(fun)
    x = ones((5,4))
    y = dfun(x)
    print y