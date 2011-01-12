from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *
import sys, warnings

hrtfdb = IRCAM_LISTEN(r'D:\HRTF\IRCAM')
hrtfset = hrtfdb.load_subject(1002)
hrtf = hrtfset(azim=30, elev=15)

def grpdelay(b, a, nfft=512, whole=False, samplerate=44.1*kHz):
    if not whole:
        nfft *= 2
    w = arange(nfft, dtype=float)/nfft*samplerate
    oa = len(a)-1 # order of a(z)
    oc = oa+len(b)-1 # order of b(z)
    c = convolve(b, a[::-1]) # c(z) = b(z)*a(1/z)*z^-oa
    cr = c*arange(oc+1, dtype=float)
    num = fft(cr, nfft)
    den = fft(c, nfft)
    minmag = 10*sys.float_info.epsilon
    polebins, = (abs(den)<minmag).nonzero()
    if len(polebins):
        warnings.warn('Group delay singular, setting to 0')
        num[b] = 0
        den[b] = 1
    gd = (num/den).real-oa
    if not whole:
        ns = nfft/2
        gd = gd[:ns]
        w = w[:ns]
    return gd, w

gd_l, w = grpdelay(asarray(hrtf.left).flatten(), array([1]), nfft=len(hrtf))
gd_r, w = grpdelay(asarray(hrtf.right).flatten(), array([1]), nfft=len(hrtf))

plot(w, (gd_l-gd_r))#/(44.1*kHz)/usecond)
show()
