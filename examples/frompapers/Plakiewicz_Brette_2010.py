#!/usr/bin/env python
'''
Spike threshold variability in single compartment model
-------------------------------------------------------
Figure 7 from:
Platkiewicz J, Brette R (2010). A Threshold Equation for Action Potential
Initiation. PLoS Comput Biol 6(7): e1000850. doi:10.1371/journal.pcbi.1000850

The original HH model is from Traub and Miles 1991, modified by Destexhe et al. 2001,
and we shift Na inactivation to -62 mV. This shift produces threshold variability,
but also spikes with variable shapes (unavoidable in a single-compartment model).

The script demonstrates that the spike threshold is proportional to the logarithm of h.
'''
from brian import *
from scipy import stats
from brian.library.electrophysiology import *

defaultclock.dt=0.05*ms
duration=500*ms
N=1000 # we simulate 1000 neurons to have more threshold statistics

rectify=lambda x:clip(x,0,inf)*siemens

# Biophysical parameters
#--Passive parameters
area=pi*(105*umetre)**2
C=1*uF
GL=0.0452*msiemens
EL=-80*mV
#--Active fixed parameters
celsius=36
temp=23
q10=2.3
#----Sodium channel parameters
ENa=50*mV
GNa=51.6*msiemens
vTraub_Na=-63*mV               #Traub convention  
vshift=-20*mV                #Inactivation shift (-62 mV instead of -42 mV)  
#--------activation
A_alpham=0.32/(ms*mV)                    #open (V) 
A_betam=0.28/(ms*mV)                    #close (V)
v12_alpham=13*mV                     #v1/2 for act                                                                     
v12_betam=40*mV                     #v1/2 for act
k_alpham=4*mV                       #act slope
k_betam=5*mV                       #act slope
#--------inactivation
A_alphah=0.128/ms                   #inact recov (V) 
A_betah=4/ms                    #inact (V)
v12_alphah=17*mV                    #v1/2 for inact                                                                     
v12_betah=40*mV                    #v1/2 for inact
k_alphah=18*mV                     #inact tau slope
k_betah=5*mV                      #inact tau slope
#----Potassium channel parameters
EK=-90*mV
#--------"delay-rectifier"
GK=10*msiemens
vTraub_K=-63*mV
A_alphan=0.032/(ms*mV)              #open (V) 
A_betan=0.5/ms                      #close (V)
v12_alphan=15*mV                    #v1/2 for act                                                                     
v12_betan=10*mV                     #v1/2 for act
k_alphan=5*mV                       #act slope
k_betan=40*mV                       #act slope
#--------muscarinic 
GK_m=0.5*msiemens
A_alphan_m=1e-4/(ms*mV)
A_betan_m=1e-4/(ms*mV)      
v12_alphan_m=-30*mV
v12_betan_m=-30*mV
k_alphan_m=9*mV
k_betan_m=9*mV  
# Input parameters
Ee=0*mV
Ei=-75*mV
taue=2.728*ms
taui=10.49*ms
Ge0=0.0121*usiemens*cm**2
Gi0=0.0573*usiemens*cm**2
Sigmae=0.012*usiemens*cm**2
Sigmai=0.0264*usiemens*cm**2
tadj=q10**((celsius-temp)/10)
ge0=Ge0/area
gi0=Gi0/area
sigmae=Sigmae/area
sigmai=Sigmai/area

Traubm=lambda v:v-vTraub_Na
alpham=lambda v:A_alpham*(Traubm(v)-v12_alpham)/(1-exp((v12_alpham-Traubm(v))/k_alpham)) 
betam=lambda v:-A_betam*(Traubm(v)-v12_betam)/(1-exp(-(v12_betam-Traubm(v))/k_betam))
minf=lambda v:alpham(v)/(alpham(v)+betam(v))
taum=lambda v:1/(alpham(v)+betam(v))
Shift=lambda v:Traubm(v)-vshift 
alphah=lambda v:A_alphah*exp((v12_alphah-Shift(v))/k_alphah) 
betah=lambda v:A_betah/(1+exp((v12_betah-Shift(v))/k_betah)) 
hinf=lambda v:alphah(v)/(alphah(v)+betah(v)) 
tauh=lambda v:1/(alphah(v)+betah(v))
TraubK=lambda v:v-vTraub_K 
alphan= lambda v:A_alphan*(TraubK(v)-v12_alphan)/(1-exp((v12_alphan-TraubK(v))/k_alphan)) 
betan= lambda v:A_betan*exp((v12_betan-TraubK(v))/k_betan)
ninf= lambda v:alphan(v)/(alphan(v)+betan(v)) 
taun= lambda v:1/(alphan(v)+betan(v))/tadj

eqs="""
dv/dt=(3*GNa*h*m**3*(ENa-v)+(GK*n**4+GK_m*n_m)*(EK-v)+GL*(EL-v)+I)/C : volt

# Sodium activation
m_inf=minf(v) : 1   #minf(v)
tau_m=taum(v) : second
dm/dt=(m_inf-m)/tau_m : 1

# Sodium inactivation
h_inf=hinf(v) : 1
tau_h=tauh(v) : second
dh/dt=(h_inf-h)/tau_h : 1

# Potassium : delay-rectifier
n_inf=ninf(v) : 1
tau_n=taun(v) : second
dn/dt=(n_inf-n)/tau_n : 1
gK=GK*n**4 : siemens

# Potassium : muscarinic
alphan_m=A_alphan_m*(v-v12_alphan_m)/(1-exp((v12_alphan_m-v)/k_alphan_m)) : hertz
betan_m=-A_alphan_m*(v-v12_alphan_m)/(1-exp(-(v12_alphan_m-v)/k_alphan_m)) : hertz
n_minf=alphan_m/(alphan_m+betan_m) : 1
taun_m=1/(alphan_m+betan_m)/tadj : second
dn_m/dt=(n_minf-n_m)/taun_m : 1
gK_m=GK_m*n_m : siemens

# Fluctuating synaptic conductances
I=rectify(ge)*(Ee-v)+rectify(gi)*(Ei-v) : amp
dge/dt=(1.5*ge0-ge)/taue+1.5*sigmae*(2./taue)**.5*xi : siemens
dgi/dt=(gi0-gi)/taui+2*sigmai*(2./taui)**.5*xi : siemens
gtot=GL+rectify(ge)+rectify(gi)+gK+gK_m : siemens
"""

neurons=NeuronGroup(N,model=eqs,implicit=True)
neurons.v=EL
neurons.m=minf(EL)
neurons.h=hinf(EL)
neurons.n=ninf(EL)
neurons.n_m=0
M=StateMonitor(neurons,'v',record=True)
Mh=StateMonitor(neurons,'h',record=True)

run(duration,report='text')

# Collect spike thresholds and values of h
threshold,logh=[],[]
valuesv,valuesh=array(M._values),array(Mh._values)
criterion=10*mV/ms # criterion for spike threshold
for i in range(N):
    v=valuesv[:,i]
    h=valuesh[:,i]
    onsets=spike_onsets(v,criterion=defaultclock.dt*criterion,vc=-35*mV)
    threshold.extend(v[onsets])
    logh.extend(-log(h[onsets]))#+log(g[onsets]))
threshold=array(threshold)/mV
logh=array(logh)/log(10.) # for display

# Voltage trace with spike onsets
subplot(311)
v=valuesv[:,0]
onsets=spike_onsets(v,criterion=defaultclock.dt*criterion,vc=-35*mV)
t=M.times[onsets]/ms
plot(M.times/ms,M[0]/mV,'k')
plot(t,v[onsets]/mV,'.r')
xlabel('t (ms)')
ylabel('V (mV)')

# Distribution of Vm and spike onsets
subplot(312)
hist(threshold,30,normed=1,histtype='stepfilled',alpha=0.6,facecolor='r')
hist(valuesv.flatten()/mV,100,normed=1,histtype='stepfilled',alpha=0.6,facecolor='k')
xlabel('V (mV)')
xlim(-80,-40)

# Relationship between h and spike threshold
subplot(313)
slope,intercept=3.1,-54. # linear regression for the prediction of threshold
p1,p2=min(logh),max(logh)
plot(logh[:len(logh)/10],threshold[:len(logh)/10],'.k') # don't show everything
plot([p1,p2],intercept+slope*array([p1,p2])*log(10.),'r')
xlabel('h')
ylabel('Threshold (mV)')
ylim(-55,-40)
xticks([0,-log(5e-1)/log(10),1,-log(5e-2)/log(10)],[1,5e-1,1e-1,5e-2])
xlim(0,1.5)

show()
