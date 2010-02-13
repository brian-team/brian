'''
Brette, R. (2008). Generation of correlated spike trains.

Response of a leaky integrate-and-fire model to correlated inputs.
Comparison with Ruben's equations.

Fig 8 in the paper.
'''

from brian import *
from brian.correlatedspikes import *
from time import time

# Firing rate of an integrate-and-fire model with white noise input current
fWNIF=lambda x: exp(x**2)*(1.+special.erf(x))
def rateIFWhiteNoise(refrac,tau,theta_eff,H_eff):
    a,_=integrate.quad(fWNIF,H_eff,theta_eff)
    return 1./(refrac+a*tau*sqrt(pi))

# Two independently correlated populations (no correlation between the two)
# ** Doubly stochastic processes **
Ne=800
Ni=200
tauc=1*ms
Je=0.04
Ji=-0.16
nu=10.*Hz
duration=20*second
c=0.001 # 0.02
taum=10.*ms
refrac=2.*ms

# Scaling
scaling=20.
nu=nu*scaling
Je=Je/sqrt(scaling)
Ji=Ji/sqrt(scaling)

'''
**********
Simulation
**********
'''
name='mixtureB'
if name=='Cox':
    # Cox processes
    input_exc=HomogeneousCorrelatedSpikeTrains(Ne,nu,c,tauc)
    input_inh=HomogeneousCorrelatedSpikeTrains(Ni,nu,c,tauc)
elif name=='mixtureA':
    # Mixture processes
    Te=mixture_process(nu=array([nu/c]),P=ones((Ne,1))*c,tauc=tauc,t=duration)
    Ti=mixture_process(nu=array([nu/c]),P=ones((Ni,1))*c,tauc=tauc,t=duration)
    input_exc=SpikeGeneratorGroup(Ne,Te)
    input_inh=SpikeGeneratorGroup(Ni,Ti)
elif name=='mixtureB':
    # Mixture processes
    Pe=eye(Ne,Ne+1)
    Pe[:,Ne]=1
    Pi=eye(Ni,Ni+1)
    Pi[:,Ni]=1
    Te=mixture_process(nu=array([c*nu]*Ne+[(1-c)*nu]),P=Pe,tauc=tauc,t=duration)
    Ti=mixture_process(nu=array([c*nu]*Ni+[(1-c)*nu]),P=Pi,tauc=tauc,t=duration)
    input_exc=SpikeGeneratorGroup(Ne,Te)
    input_inh=SpikeGeneratorGroup(Ni,Ti)

neuron=NeuronGroup(1,model='dv/dt=-v/taum : 1',reset=0,threshold=1,refractory=refrac)
exc_synapses=Connection(input_exc,neuron,'v')
exc_synapses.connect_full(input_exc,neuron,Je)
inh_synapses=Connection(input_inh,neuron,'v')
inh_synapses.connect_full(input_inh,neuron,Ji)
counter=SpikeCounter(neuron)

t1=time()
run(duration)
t2=time()
print "Simulated in",t2-t1,"s"
print "Rate:",counter[0]/duration

'''
***********************
Theoretical predictions
***********************
'''
# Ruben's formulas
F=c+1. # Fano factor
rho=c/F # Spike count correlation
sigmaw2=Je*Je*Ne*nu+Ji*Ji*Ni*nu
sigma2=Je*Je*nu*Ne*(F-1.+(Ne-1.)*F*rho)+Ji*Ji*nu*Ni*(F-1.+(Ni-1.)*F*rho)

alpha=sigma2/sigmaw2 # correlation magnitude
mu=Je*Ne*nu+Ji*Ni*nu
sigmaeff=sqrt(sigmaw2+sigma2)
sigmaw=sqrt(sigmaw2)
thetahat=(1.-mu*taum)/(sigmaw*sqrt(taum))
Hhat=-mu*taum/(sigmaw*sqrt(taum))
    
print "Fano factor =",F
print "Spike count correlation (rho) =",rho
print "Correlation magnitude (alpha) =",alpha
print "Mean input =",mu
print "Input variance =",sigma2

nueff=rateIFWhiteNoise(refrac,taum,(1.-mu*taum)/(sigmaeff*sqrt(taum)),-mu*taum/(sigmaeff*sqrt(taum)))
nu0=rateIFWhiteNoise(refrac,taum,thetahat,Hhat)

print "Theoretical rate for uncorrelated inputs =",nu0
print "Theoretical rate for instantaneous correlations =",nueff

# coefficient for Ruben's long tauc formula
C=alpha*(taum*nu0)**2*(taum*nu0*(fWNIF(thetahat)-fWNIF(Hhat))**2/(1.-nu0*refrac)-\
            (thetahat*fWNIF(thetahat)-Hhat*fWNIF(Hhat))/sqrt(2))

print "C =",C

shortprediction=nueff-alpha*nu0*nu0*sqrt(.5*pi*taum*tauc)*fWNIF((1.-mu*taum)/(sigmaw*sqrt(taum)))
longprediction=nu0+C/tauc

print "Predicted rate for short tauc:",shortprediction
print "Predicted rate for long tauc:",longprediction
