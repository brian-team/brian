'''
Random processes for Brian.
'''
from brian.equations import Equations
from brian.units import get_unit
from scipy.signal import lfilter
from numpy import exp, sqrt
from numpy.random import randn
from brian.stdunits import ms

__all__ = ['OrnsteinUhlenbeck', 'white_noise', 'colored_noise']

def OrnsteinUhlenbeck(x, mu, sigma, tau):
    '''
    An Ornstein-Uhlenbeck process.
    mu = mean
    sigma = standard deviation
    tau = time constant
    x = name of the variable
    Returns an Equations() object
    '''
    return Equations('dx/dt=(mu-x)*invtau+sigma*((2.*invtau)**.5)*xi : unit', \
                     x=x, mu=mu, sigma=sigma, invtau=1. / tau, unit=get_unit(mu))


def white_noise(dt, duration):
    n = int(duration/dt)
    noise = randn(n)
    return noise

def colored_noise(tau, dt, duration):
    noise = white_noise(dt, duration)
    a = [1., -exp(-dt/tau)]
    b = [1.]
    fnoise = sqrt(2*dt/tau)*lfilter(b, a, noise)
    return fnoise

if __name__ == '__main__':
    tau = 10*ms
    dt = .1*ms
    duration = 1000*ms
    noise = white_noise(dt, duration)
    cnoise = colored_noise(tau, dt, duration)
    from numpy import linspace
    t = linspace(0*ms, duration, len(noise))
    
    from pylab import acorr, show, subplot, plot, ylabel, xlim
    
    subplot(221)
    plot(t, noise)
    ylabel('white noise')
    
    subplot(222)
    acorr(noise, maxlags=200)
    xlim(-200,200)
    
    subplot(223)
    plot(t, cnoise)
    ylabel('colored noise')
    
    subplot(224)
    acorr(cnoise, maxlags=200)
    xlim(-200,200)
    
    show()