from brian import *
from matplotlib import rc
rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex=True)

#Parameters from Kuznetsov et al. (2006) Table 1
Mg=1.4*mmole
F=96485.3999*coulomb/mole
E_Ca=100*mV
g_L=0.05*msiemens/cm2
g_maxCa=0.2*msiemens/cm2
g_maxK=0.4*msiemens/cm2
g_maxKCa=0.3*msiemens/cm2
g_AMPA=0.07*msiemens/cm2 #From Figure 5 Kuznetsov et al. (2006)
g_maxNMDA=1*msiemens/cm2 #From Figure 5 Kuznetsov et al. (2006)
E_k=-90*mV
E_L=-50*mV
E_AMPA=0*mV
E_NMDA=0*mV
p_Ca=2500*um/second
beta=0.05 # ratio of free to total calcium
z=2 # Valence of Calcium Ion
C=1*uF/cm2
k=250*nmole/cm2
r=1*um
eqs=Equations('''
    dV/dt = (g_Ca*(E_Ca-V) + g_K *(E_k-V) + g_KCa*(E_k-V) + g_L*(E_L-V) + g_NMDA*(E_NMDA-V)+g_AMPA*(E_AMPA-V)+ I)/C: volt
    dCa/dt = (4*beta)*(g_Ca/(z*F)*(E_Ca-V) - p_Ca*Ca/r) : nmole/cm2
    g_KCa = g_maxKCa * Ca**4/(Ca**4 + k**4) : msiemens/cm2
    alpha_Ca = -0.0032*(V+50*mV)/((exp(-(V+50*mV)/(5*mV))-1)*mV) : 1
    beta_Ca = 0.05*exp(-(V+55*mV)/(40*mV)) : 1
    g_Ca = g_maxCa*(alpha_Ca/(alpha_Ca + beta_Ca))**4 : usiemens/cm2
    g_NMDA = g_maxNMDA/(1+Mg/(10*mmole)*exp(-V/(12.5*mV))) : usiemens/cm2
    g_K = g_maxK/(1+exp(-(V+10*mV)/(7*mV)))    : usiemens/cm2
    I : uA/cm2

    ''')
test=NeuronGroup(1, model=eqs, threshold=-40*mV, reset=-53*mV)
trace=StateMonitor(test, 'V', record=True)
stimulus=StateMonitor(test, 'I', record=True)
test.V=-50*mV
run(1000*msecond)
test.I=1*uA/cm**2
run(500*msecond)
test.I=0*uA/cm**2
run(500*msecond)

figure()
title('Simulation')
subplot(211)
plot(trace.times/ms, trace[0]/mV)
ylabel('Voltage (mV)')
subplot(212)
plot(stimulus.times/ms, stimulus[0]/uA*cm2)
ylabel(r'\Current Density ($\frac{\mu A}{cm^{2}}$)')
xlabel('Time (ms)')
show()

