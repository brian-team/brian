'''
Random processes for Brian.
'''
from brian.equations import Equations
from brian.units import get_unit

__credits__=dict(author    = 'Romain Brette (brette@di.ens.fr)',
                 date      = 'April 2008')

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
