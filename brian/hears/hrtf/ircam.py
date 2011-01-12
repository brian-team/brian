from brian import *
from scipy.io import loadmat # NOTE: this requires scipy 0.7+
from glob import glob
from copy import copy
from scipy.io.wavfile import *
import os
from hrtf import *

__all__ = ['IRCAM_LISTEN']

class IRCAM_LISTEN(HRTFDatabase):
    '''
    :class:`HRTFDatabase` for the IRCAM LISTEN public HRTF database.
    
    For details on the database, see the
    `website <http://recherche.ircam.fr/equipes/salles/listen/>`__.
    
    The database object can be initialised with the following arguments:
    
    ``basedir``
        The directory where the database has been downloaded and extracted,
        e.g. ``r'D:\HRTF\IRCAM'``.
    ``compensated=False``
        Whether to use the raw or compensated impulse responses.
    ``samplerate=None``
        If specified, you can resample the impulse responses to a different
        samplerate, otherwise uses the default 44.1 kHz.
    
    The coordinates are pairs ``(azim, elev)`` where ``azim`` ranges from 0
    to 345 degrees in steps of 15 degrees, and elev ranges from -45 to 90 in
    steps of 15 degrees.
    
    **Obtaining the database**
    
    The database can be downloaded
    `here <http://recherche.ircam.fr/equipes/salles/listen/download.html>`__.
    Each subject archive should be extracted to a folder (e.g. IRCAM) with the
    names of the subject, e.g. IRCAM/IRC_1002, etc.
    '''
    def __init__(self, basedir, compensated=False, samplerate=None):
        self.basedir = basedir
        self.compensated = compensated
        names = glob(os.path.join(basedir, 'IRC_*'))
        splitnames = [os.path.split(name) for name in names]

        self.subjects = [int(name[4:]) for base, name in splitnames]
        if samplerate is not None:
            raise ValueError('Custom samplerate not supported.')
        self.samplerate = samplerate

    def load_subject(self, subject):
        subject = str(subject)
        filename = os.path.join(self.basedir, 'IRC_' + subject)
        if self.compensated:
            filename = os.path.join(filename, 'COMPENSATED/MAT/HRIR/IRC_' + subject + '_C_HRIR.mat')
        else:
            filename = os.path.join(filename, 'RAW/MAT/HRIR/IRC_' + subject + '_R_HRIR.mat')
        samplerate = 44.1*kHz
        m = loadmat(filename, struct_as_record=True)
        if 'l_hrir_S' in m.keys(): # RAW DATA
            affix = '_hrir_S'
        else:                      # COMPENSATED DATA
            affix = '_eq_hrir_S'
        l, r = m['l' + affix], m['r' + affix]
        azim = l['azim_v'][0][0][:, 0]
        elev = l['elev_v'][0][0][:, 0]
        coords = make_coordinates(azim=azim, elev=elev)
        l = l['content_m'][0][0]
        r = r['content_m'][0][0]
        # self.data has shape (num_ears=2, num_indices, hrir_length)
        data = vstack((reshape(l, (1,) + l.shape), reshape(r, (1,) + r.shape)))
        hrtfset = HRTFSet(data, samplerate, coords)
        hrtfset.name = 'IRCAM_'+subject
        return hrtfset
