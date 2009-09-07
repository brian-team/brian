'''
Random processes for Brian.
'''
from brian.equations import Equations
from brian.units import get_unit

def OrnsteinUhlenbeck(x,mu,sigma,tau):
    '''
    An Ornstein-Uhlenbeck process.
    mu = mean
    sigma = standard deviation
    tau = time constant
    x = name of the variable
    Returns an Equations() object
    '''
    return Equations('dx/dt=(mu-x)*invtau+sigma*((2.*invtau)**.5)*xi : unit',\
                     x=x,mu=mu,sigma=sigma,invtau=1./tau,unit=get_unit(mu))
