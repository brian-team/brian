from brian import *
from scipy.io import loadmat # NOTE: this requires scipy 0.7+
from glob import glob
from copy import copy
from scipy.io.wavfile import *
import os
from hrtf import *
from coordinates import *

__all__ = ['IRCAM_HRTFSet', 'IRCAM_LISTEN']

class IRCAM_HRTFSet(HRTFSet):
    '''
    TODO: documentation
    '''
    def load(self, filename, samplerate=None, coordsys=None, name=None):
        # TODO: check samplerate
        if name is None:
            _, name = os.path.split(filename)
        self.name = name
        m = loadmat(filename, struct_as_record=True)
        if 'l_hrir_S' in m.keys(): # RAW DATA
            affix = '_hrir_S'
        else:                      # COMPENSATED DATA
            affix = '_eq_hrir_S'
        l, r = m['l' + affix], m['r' + affix]
        self.azim = l['azim_v'][0][0][:, 0]
        self.elev = l['elev_v'][0][0][:, 0]
        l = l['content_m'][0][0]
        r = r['content_m'][0][0]
        coords = AzimElevDegrees.make(len(self.azim))
        coords['azim'] = self.azim
        coords['elev'] = self.elev
        if coordsys is not None:
            self.coordinates = coords.convert_to(coordsys)
        else:
            self.coordinates = coords
        # self.data has shape (num_ears=2, num_indices, hrir_length)
        self.data = vstack((reshape(l, (1,) + l.shape), reshape(r, (1,) + r.shape)))
        self.samplerate = 44.1 * kHz


class IRCAM_LISTEN(HRTFDatabase):
    def __init__(self, basedir, compensated=False, samplerate=None):
        self.basedir = basedir
        self.compensated = compensated
        names = glob(os.path.join(basedir, 'IRC_*'))
        splitnames = [os.path.split(name) for name in names]

        self.subjects = [int(name[4:]) for base, name in splitnames]
        self.samplerate = samplerate

    def subject_name(self, subject):
        return 'IRCAM_' + str(subject)

    def load_subject(self, subject):
        subject = str(subject)
        fname = os.path.join(self.basedir, 'IRC_' + subject)
        if self.compensated:
            fname = os.path.join(fname, 'COMPENSATED/MAT/HRIR/IRC_' + subject + '_C_HRIR.mat')
        else:
            fname = os.path.join(fname, 'RAW/MAT/HRIR/IRC_' + subject + '_R_HRIR.mat')
        return IRCAM_HRTFSet(fname, samplerate=self.samplerate, name=self.subject_name(subject))
