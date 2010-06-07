from brian import *
from customrefractoriness import *
from scipy.optimize import fmin
from brian.tools.autodiff import *

#----Fixed parameters
ENa=60*mV
El=-70*mV
tau=5*ms
tauh=5*ms

#----Input parameters
mu=0*mV
sigma=2*15*mV
tauc=10*ms
k=1/3*sigma/ms                                                      #empirical threshold criterion

#----Na channels parameters
#--load the data
#Va,Ka,Vi,Ki,R=load('Nav1_6.txt').T#I don't have this file
Va, Ka, Vi, Ki, R=(zeros(10), zeros(10), zeros(10), zeros(10), zeros(10))
N=len(Va)

eqs=Equations("""
# Sodium channel model
Pa=1/(1+exp(-(v-va)/ka)) : 1
Pi=1/(1+exp(-(v-vi)/ki)) : 1

# Membrane equation
dv/dt=(-r*Pa*h*(v-ENa)-(v-El)+I)/tau : volt
dh/dt=((1-Pi)-h)/tauh : 1

# Input
dI/dt=(mu-I)/tauc+sigma*(.5*tauc)**-.5*xi : volt

# Sodium channel model Parameters
va : mV
ka : mV
vi : mV
ki : mV
r : 1
""")
eqs.prepare()

#--Function for computing the EIF and threshold dynamics parameters
def threshold_parameters(f, h_inf, rest=El):
   '''
   Calculates the threshold and the slope factor from the
   equation tau*dv/dt=f(v)
   '''
   vt=fmin(f, rest, disp=False)[0] # Threshold (simplex algorithm)
   vt=vt*volt
   deltat=1/(tau*differentiate(f, vt, order=2)) # Slope factor
   a=-differentiate(h_inf, vt, order=1)*deltat/h_inf(vt)
   b=-deltat*math.log(h_inf(vt))+(1-a)*vt
   return (vt, deltat, a, b)

#--Predictions for each Na channel model
vt=[[] for i in range(len(Va))]
deltat=[[] for i in range(len(Va))]
a=[[] for i in range(len(Va))]
b=[[] for i in range(len(Va))]
pente=[[] for i in range(len(Va))]
ordonnee=[[] for i in range(len(Va))]
figure()
for j in range(len(Va)):
    print "Sodium channel model", j+1
    print "----------------------"
    va=Va[j]*mV
    ka=Ka[j]*mV
    vi=Vi[j]*mV
    ki=Ki[j]*mV
    r=R[j]
    print "Va =", va, "mV ; ka =", ka, "mV ;", "Vi =", vi, "mV ; ki =", ki, "mV"
#--Boltzmann functions
    Pa=lambda v: eqs.apply('Pa', {'v':v, 'va':va, 'ka':ka})                            # activation
    h_inf=lambda v: 1-eqs.apply('Pi', {'v':v, 'vi':vi, 'ki':ki})                           # inactivation
#--The EIF model   
    # F=lambda v: eqs.apply('v',{'Pa':Pa(v),'r':r,'h':1,'I':mu,'v':v})  # TODO: Romain, this doesn't work and probably should, see note in Equations.apply()
    F=lambda v: eqs.apply('v', {'r':r, 'h':1, 'I':mu, 'v':v, 'va':va, 'ka':ka})
    v=linspace(-100*mV, 0*mV, 200)
    plot(v/mV, Pa(v))
    plot(v/mV, F(v)*ms/mV)
show()
