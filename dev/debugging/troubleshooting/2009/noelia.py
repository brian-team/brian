from pylab import *
from scipy import integrate

# parameters
area=0.0002
Cm=(1)*area
El=-63.8
EK=-90
ENa=55
gl=(0.1025)*area
g_na=(35)*area
g_kd=(9)*area
Iapp=0.001


def dV_dt(V, t=0):

    alpham=0.1*(V[0]+35.)/(1.-exp(-(V[0]+35.)/10.))
    betam=4.*exp(-(V[0]+60.)/18.)
    alphah=0.07*exp(-(V[0]+58.)/20.)
    betah=1./(1.+exp(-(V[0]+28.)/10.))
    alphan=0.05*(V[0]+34)/(1.-exp(-(V[0]+34)/10))
    betan=0.625*exp(-(V[0]+44)/80)
    minf=alpham/(alpham+betam)
    return array([(gl*(El-V[0])-g_na*(minf)**3*V[1]*(V[0]-ENa)-g_kd*(V
[2]**4)*(V[0]-EK)+Iapp)/Cm,
                   5*(alphah*(1-V[1])-betah*V[1]),
                   alphan*(1-V[2])-betan*V[2] ])


V0=([-65, 0, 0])

t=linspace(0, 100, 100000)

print " running..."

V, infodict=integrate.odeint(dV_dt, V0, t, full_output=True)

infodict['message']

plot(t, V[:, 0], 'r-')

show()
