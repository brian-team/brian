from brian import *
import brianlib as bl
import time
####### Parameters ############################
domonitorandplot = False
debugmode = False
duration = 2.5*second
we = 1.62*mV # 1.62*mV for low firing rate, 9*mV for high
N = 100
####### Definition of CUBA network ############
Nsyn = 20.
eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
Ne = int(N*0.8)
Ni = N-Ne
P=NeuronGroup(N,model=eqs,
              threshold=-50*mV,reset=-60*mV, refractory=5*ms)
Pe=P.subgroup(Ne)
Pi=P.subgroup(Ni)
if domonitorandplot:
    Pe.spikemonitor = SpikeMonitor(Pe)
    Pi.spikemonitor = SpikeMonitor(Pi)
    P.spikemonitor = SpikeMonitor(P)
Ce=Connection(Pe,P,'ge',structure='dense')
Ci=Connection(Pi,P,'gi',structure='dense')    
cp = Nsyn/N    
Ce.connect_random(Pe, P, cp, weight=we)
Ci.connect_random(Pi, P, cp, weight=-9*mV)
if domonitorandplot:
    M = StateMonitor(P, 'v', record=True)
P.v = -60*mV+10*mV*rand(len(P))
P.ge = we*rand(len(P))
P.gi = -9*mV*rand(len(P))
####### Generate cBrian version ###############
c = array(P._state_updater._C.flatten()) # if we don't do this the memory is corrupted
blPsu = bl.LinearStateUpdater(P._state_updater.A, c)
if debugmode:
    print 'cBrian state updater initialised OK'
blPthr = bl.Threshold(P._threshold.state, float(P._threshold.threshold))
if debugmode:
    print 'cBrian threshold initialised OK'
period = int(P._resetfun.period/P.clock.dt)+1
blPreset = bl.Refractoriness(P._resetfun.state, float(P._resetfun.resetvalue), period)
if debugmode:
    print 'cBrian reset initialised OK'
blP = bl.NeuronGroup(P._S, blPsu, blPthr, blPreset, P.LS.S.n, P.LS.ind.n)
if debugmode:
    print 'cBrian P initialised OK'
blPe = bl.NeuronGroup(blP, Pe._origin, len(Pe))
if debugmode:
    print 'cBrian Pe initialised OK'
blPi = bl.NeuronGroup(blP, Pi._origin, len(Pi))
if debugmode:
    print 'cBrian Pi initialised OK'
blCe_W = bl.DenseConnectionMatrix(Ce.W)
if debugmode:
    print 'cBrian Ce_W initialised OK'
blCi_W = bl.DenseConnectionMatrix(Ci.W)
if debugmode:
    print 'cBrian Ci_w initialised OK'
blCe = bl.Connection(blPe, blP, blCe_W, P.get_var_index('ge'))
if debugmode:
    print 'cBrian Ce initialised OK'
blCi = bl.Connection(blPi, blP, blCi_W, P.get_var_index('gi'))
if debugmode:
    print 'cBrian Ci initialised OK'
if domonitorandplot:
    blM = bl.StateMonitor(blP, 0)
    if debugmode:
        print 'cBrian M initialised OK'
blnet = bl.Network()
if debugmode:
    print 'cBrian net initialised OK'
blnet.add(blP)
if debugmode:
    print 'cBrian P added to net OK'
blnet.add(blCe)
if debugmode:
    print 'cBrian Ce added to net OK'
blnet.add(blCi)
if debugmode:
    print 'cBrian Ci added to net OK'
if domonitorandplot:
    blnet.add(blM)
    if debugmode:
        print 'cBrian M added to net OK'
####### Test cBrian in debugmode ##############
#if debugmode:
#    blP.update()
#    print 'debugmode cBrian blP.update() OK'
#    blP.reset()
#    print 'debugmode cBrian blP.reset() OK'
####### Copy data #############################
V_start = array(P.v, copy=True)
ge_start = array(P.ge, copy=True)
gi_start = array(P.gi, copy=True)
####### Run cBrian version ####################
P.v = V_start
P.ge = ge_start
P.gi = gi_start
start = time.time()
if domonitorandplot:
    blsm_P = []
    blsm_Pe = []
    blsm_Pi = []
    from itertools import repeat
    for t in [i*defaultclock.dt for i in range(int(duration/defaultclock.dt))]:
        blnet.update()
        blsm_P.extend(zip(blP.get_spikes(), repeat(t)))
        blsm_Pe.extend(zip(blPe.get_spikes(), repeat(t)))
        blsm_Pi.extend(zip(blPi.get_spikes(), repeat(t)))
else:
    blnet.run(int(duration/defaultclock.dt))
print 'cBrian:', (time.time()-start)*second
V_after_cBrian = array(P.v, copy=True)
####### Run Brian version #####################
P.v = V_start
P.ge = ge_start
P.gi = gi_start
start = time.time()
run(duration)
print 'Brian:', (time.time()-start)*second
V_after_Brian = array(P.v, copy=True)
####### Compare ###############################
print 'max abs diff of V values at end of sim:', max(abs(V_after_Brian-V_after_cBrian))
if domonitorandplot:
    subplot(211)
    for i in range(10):
        plot(blM[i])
    title('cBrian')
    subplot(212)
    for i in range(10):
        plot(M[i])
    title('Brian')
    figure()
    subplot(311)
    raster_plot(P.spikemonitor)
    subplot(312)
    raster_plot(Pe.spikemonitor)
    subplot(313)
    raster_plot(Pi.spikemonitor)
    subplot(311)
    if len(blsm_P):
        i, t = zip(*blsm_P)
        plot(array(t)/ms, i, '+')
    subplot(312)
    if len(blsm_Pe):
        i, t = zip(*blsm_Pe)
        plot(array(t)/ms, i, '+')
    subplot(313)
    if len(blsm_Pi):
        i, t = zip(*blsm_Pi)
        plot(array(t)/ms, i, '+')
    show()