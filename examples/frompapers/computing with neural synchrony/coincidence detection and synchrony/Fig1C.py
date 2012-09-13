#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Timescale and strength
"""
from brian import *

rc('lines',linewidth=2)
rc('font',size=16)
rc('xtick',labelsize=16)
rc('ytick',labelsize=16)
rc('legend',fontsize=16)
rc('axes',labelsize=16,titlesize=16)
w,h=rcParamsDefault['figure.figsize']
fontsize=16

SNR,timescale,strength=loadtxt('reproducibility.txt')
SNR=10*log(SNR)/log(10)
#SNR=SNR[1:]
#timescale=timescale[1:]
#strength=strength[1:]

figure(figsize=(w*1.5,h*.5))
subplot(121)
plot(SNR,timescale/ms,'k')
#plot(SNR,timescale/ms,'r.')
plot(SNR,0*SNR+7,'k--')
xlabel('SNR (dB)')
ylabel('Precision (ms)')
subplot(122)
plot(SNR,strength*100,'k')
ylim(0,100)
#plot(SNR,strength*100,'r.')
xlabel('SNR (dB)')
ylabel('Reliability (%)')
savefig('FigC.eps')
show()
