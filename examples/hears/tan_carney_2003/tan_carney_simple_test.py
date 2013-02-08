'''
Fig. 1 and 3 (spking output without spiking/refractory period) should
reproduce the output of the AN3_test_tone.m and AN3_test_click.m
scripts, available in the code accompanying the paper Tan & Carney (2003).
This matlab code is available from
http://www.urmc.rochester.edu/labs/Carney-Lab/publications/auditory-models.cfm

Tan, Q., and L. H. Carney.
"A Phenomenological Model for the Responses of Auditory-nerve Fibers.
II. Nonlinear Tuning with a Frequency Glide".
The Journal of the Acoustical Society of America 114 (2003): 2007.
'''

import numpy as np
import matplotlib.pyplot as plt

from brian.stdunits import kHz, Hz, ms
from brian.network import Network
from brian.monitor import StateMonitor, SpikeMonitor
from brian.globalprefs import set_global_preferences

#set_global_preferences(useweave=True)
from brian.hears import (Sound, get_samplerate, set_default_samplerate, tone,
                         click, silence, dB, TanCarney, MiddleEar, ZhangSynapse)
from brian.clock import reinit_default_clock


set_default_samplerate(50*kHz)
sample_length = 1 / get_samplerate(None)
cf = 1000 * Hz

print 'Testing click response'
duration = 25*ms    
levels = [40, 60, 80, 100, 120]
# a click of two samples
tones = Sound([Sound.sequence([click(sample_length*2, peak=level*dB),
                               silence(duration=duration - sample_length)])
           for level in levels])
ihc = TanCarney(MiddleEar(tones), [cf] * len(levels), update_interval=1)
syn = ZhangSynapse(ihc, cf)
s_mon = StateMonitor(syn, 's', record=True, clock=syn.clock)
R_mon = StateMonitor(syn, 'R', record=True, clock=syn.clock)
spike_mon = SpikeMonitor(syn)
net = Network(syn, s_mon, R_mon, spike_mon)
net.run(duration * 1.5)
for idx, level in enumerate(levels):
    plt.figure(1)
    plt.subplot(len(levels), 1, idx + 1)
    plt.plot(s_mon.times / ms, s_mon[idx])
    plt.xlim(0, 25)
    plt.xlabel('Time (msec)')
    plt.ylabel('Sp/sec')
    plt.text(15, np.nanmax(s_mon[idx])/2., 'Peak SPL=%s SPL' % str(level*dB));
    ymin, ymax = plt.ylim()
    if idx == 0:
        plt.title('Click responses')

    plt.figure(2)
    plt.subplot(len(levels), 1, idx + 1)
    plt.plot(R_mon.times / ms, R_mon[idx])
    plt.xlabel('Time (msec)')
    plt.xlabel('Time (msec)')
    plt.text(15, np.nanmax(s_mon[idx])/2., 'Peak SPL=%s SPL' % str(level*dB));
    plt.ylim(ymin, ymax)
    if idx == 0:
        plt.title('Click responses (with spikes and refractoriness)')
    plt.plot(spike_mon.spiketimes[idx] / ms,
         np.ones(len(spike_mon.spiketimes[idx])) * np.nanmax(R_mon[idx]), 'rx')

print 'Testing tone response'
reinit_default_clock()
duration = 60*ms    
levels = [0, 20, 40, 60, 80]
tones = Sound([Sound.sequence([tone(cf, duration).atlevel(level*dB).ramp(when='both',
                                                                         duration=10*ms,
                                                                         inplace=False),
                               silence(duration=duration/2)])
               for level in levels])
ihc = TanCarney(MiddleEar(tones), [cf] * len(levels), update_interval=1)
syn = ZhangSynapse(ihc, cf)
s_mon = StateMonitor(syn, 's', record=True, clock=syn.clock)
R_mon = StateMonitor(syn, 'R', record=True, clock=syn.clock)
spike_mon = SpikeMonitor(syn)
net = Network(syn, s_mon, R_mon, spike_mon)
net.run(duration * 1.5)
for idx, level in enumerate(levels):
    plt.figure(3)
    plt.subplot(len(levels), 1, idx + 1)
    plt.plot(s_mon.times / ms, s_mon[idx])
    plt.xlim(0, 120)
    plt.xlabel('Time (msec)')
    plt.ylabel('Sp/sec')
    plt.text(1.25 * duration/ms, np.nanmax(s_mon[idx])/2., '%s SPL' % str(level*dB));
    ymin, ymax = plt.ylim()
    if idx == 0:
        plt.title('CF=%.0f Hz - Response to Tone at CF' % cf)

    plt.figure(4)
    plt.subplot(len(levels), 1, idx + 1)
    plt.plot(R_mon.times / ms, R_mon[idx])
    plt.xlabel('Time (msec)')
    plt.xlabel('Time (msec)')
    plt.text(1.25 * duration/ms, np.nanmax(R_mon[idx])/2., '%s SPL' % str(level*dB));
    plt.ylim(ymin, ymax)
    if idx == 0:
        plt.title('CF=%.0f Hz - Response to Tone at CF (with spikes and refractoriness)' % cf)
    plt.plot(spike_mon.spiketimes[idx] / ms,
         np.ones(len(spike_mon.spiketimes[idx])) * np.nanmax(R_mon[idx]), 'rx')

plt.show()
