def importSpkTrain(Nout,file_name="embeddedInputs.mat",var="embeddedInputs"):
    global InhNeur, ExNeur, SpkTrainEx, SpkTrainInh
    cellImport = scipy.io.loadmat(file_name, matlab_compatible=True)
    inputSpkTimes = cellImport[var]
        
    allSpkTrain=[]
    allSpkTrainImport=[]
    for j in range(Nout):
        spkTimeTemp=[]
        for ind1 in range(len(inputSpkTimes[j,inputInstance-1])):
            spkTimeTemp.append(Quantity(inputSpkTimes[j,inputInstance-1][ind1][0])*ms)
    
        allSpkTrain.append(spkTimeTemp)
    
    shuffleNeur=random.permutation(Nout) # implement 12.5 or 25 % of inhibition
    #InhNeur = shuffleNeur[:Nout/4]
    InhNeur = shuffleNeur[:Nout/16] # 16
    #ExNeur =  shuffleNeur[Nout/4:len(shuffleNeur)]
    ExNeur =  shuffleNeur[Nout/16:len(shuffleNeur)] #16
    
    SpkTrainEx=[]
    for indEx in range(len(ExNeur)):
        SpkTrainEx.append(allSpkTrain[ExNeur[indEx]])
        
    SpkTrainInh=[]
    for indInh in range(len(InhNeur)):
        SpkTrainInh.append(allSpkTrain[InhNeur[indInh]])
    
    return InhNeur, ExNeur, SpkTrainEx, SpkTrainInh

def Wdefinition(Nout,ExNeur,InhNeur):
    global W_ex, W_inh
    W=ones((Nout,Nout))  
    W_ex=W[:len(ExNeur),:]
    W_inh=W[len(ExNeur):,:]
    
    return W_ex, W_inh

def importDelays(file_name2="delta_ij.mat",var2="delta_ij"):
    global delta_ij
    d = scipy.io.loadmat(file_name2, matlab_compatible=True, mdict=None)
    delta_ij=d[var2]
    
    return delta_ij

#-------------------------------------------------------------------------------
from brian import *
from numpy import *
from scipy import *
import scipy.io
from scipy.stats import poisson
from numpy.random import *

simclock = Clock(dt=0.001 *ms)
monclock = Clock(dt=0.1 *ms)

patch_size = 8 #

numPix=patch_size*patch_size
Nout = numPix

inputInstance = 3       #  

C_m = 0.5 * nF
tau_m = 20 * ms

Vt = -50 * mV		# spike threshold | 1 * mV
Vr = -60 * mV		# resting potential and reset value | 0 * mV
stdVr = 3               # std of Vr

E_L = -70 * mV          # resting potential | 0 * mV
gL = 100                # leak conductance 100 * nS
     
E_ex = 0 * mV           # excitatory reverse potential
#g_ex = 25 * nS
tau_ex = 1 * ms         # 5 * ms

E_inh = -70 * mV        # inhibitory reverse potential
#g_inh = 20 * nS
tau_inh = 1 * ms       # 10 * mS

ref = 3 * ms            # neurons' refractory period

# COBA model
eqs = Equations('''
        dV/dt = (-gL*(V-E_L)-g_ex*(V-E_ex)-g_inh*(V-E_inh))*(1./C_m)    : volt
        dg_ex/dt = -g_ex/tau_ex                                         : nS
        dg_inh/dt = -g_inh/tau_inh                                      : nS
''')

#eqs = Equations('''
#        dV/dt = (-gL*(V-E_L)-g_ex*(V-E_ex)-g_inh*(V-E_inh))*(1./tau_m)    : volt
#        dg_ex/dt = -g_ex/tau_ex                                           : 1
#        dg_inh/dt = -g_inh/tau_inh                                        : 1
#''')


SYN_PROB=0.5 #0.2

def f(spikes):
    #print spikes.shape
    if (spikes.shape[0])>0:
        prob=scipy.stats.bernoulli.rvs(SYN_PROB,size=spikes.shape[0])
        spikes*=prob

 
import random
from numpy import *
importSpkTrain(Nout)

Input_ex = MultipleSpikeGeneratorGroup(SpkTrainEx, clock=simclock)
Input_inh = MultipleSpikeGeneratorGroup(SpkTrainInh, clock=simclock)

OutLayer = NeuronGroup(N=Nout,  model=eqs, threshold=Vt, reset=Vr, refractory=ref, clock=simclock) #, freeze=False, implicit=True) 

Wdefinition(Nout,ExNeur,InhNeur)

importDelays()

C_ex = Connection(Input_ex,OutLayer,'g_ex',delay=True,max_delay=delta_ij.max()*ms)
C_inh = Connection(Input_inh,OutLayer,'g_inh',delay=True,max_delay=delta_ij.max()*ms)
  
for ej,ei in ((ej,ei) for ej in range(len(ExNeur)) for ei in range(Nout)):
    C_ex.delay[ej,ei] = Quantity(delta_ij[ExNeur[ej],inputInstance-1])*ms

C_ex.connect(Input_ex,OutLayer,W_ex)

for ij,ii in ((ij,ii) for ij in range(len(InhNeur)) for ii in range(Nout)):
    C_inh.delay[ij,ii] = Quantity(delta_ij[InhNeur[ij],inputInstance-1])*ms

C_inh.connect(Input_inh,OutLayer,W_inh)

SMonitor = []
SMonitor.append(SpikeMonitor(OutLayer, function=f))

OutLayer.V = Vr+(Vt-Vr)*rand(len(OutLayer)) # uniform random initialization of the  membrane voltage V of the Output layer between Vr and Vt
#OutLayer.V = random.normal(Vr,stdVr,len(OutLayer))

# Monitors ---------------------------------------------------------------------
I_ex = SpikeMonitor(Input_ex)
I_inh = SpikeMonitor(Input_inh)
M = SpikeMonitor(OutLayer,record=True)
outVolt = StateMonitor(OutLayer,'V', record=True, clock=simclock)
outGex = StateMonitor(OutLayer,'g_ex', record=True, clock=monclock)
outGinh = StateMonitor(OutLayer,'g_inh', record=True, clock=monclock)

# run and plot -----------------------------------------------------------------
run(0.05 * second)

figure()
raster_plot(M)
show()
xlabel('time (ms)')
ylabel('neuron number')
show()

#raster_plot(I_ex)
#raster_plot(I_inh)
#show()
#
figure()
plot(outVolt.times / ms, outVolt[0] / mV)
show()

figure()
plot(outGex.times, outGex[0])
plot(outGinh.times, outGinh[0])
show()
