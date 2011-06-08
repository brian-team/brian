from brian import *
from scipy.io import loadmat # NOTE: this requires scipy 0.7+
from glob import glob
from copy import copy
from scipy.io.wavfile import *
import os, re
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
        e.g. ``r'D:\HRTF\IRCAM'``. Multiple directories in a list can be provided as well (e.g IRCAM and IRCAM New).
    ``compensated=False``
        Whether to use the raw or compensated impulse responses.
    ``samplerate=None``
        If specified, you can resample the impulse responses to a different
        samplerate, otherwise uses the default 44.1 kHz.
    
    The coordinates are pairs ``(azim, elev)`` where ``azim`` ranges from 0
    to 345 degrees in steps of 15 degrees, and elev ranges from -45 to 90 in
    steps of 15 degrees. After loading the database, the attribute 'subjects' gives all the subjects number that were detected as installed.
    
    **Obtaining the database**
    
    The database can be downloaded
    `here <http://recherche.ircam.fr/equipes/salles/listen/download.html>`__.
    Each subject archive should be extracted to a folder (e.g. IRCAM) with the
    names of the subject, e.g. IRCAM/IRC_1002, etc.
    '''
    def __init__(self, basedir, compensated=False, samplerate=None):
        if not isinstance(basedir, (list, tuple)):
            basedir = [basedir]
        self.basedir = basedir
        self.compensated = compensated
        names = []
        for basedir in self.basedir:
            names += glob(os.path.join(basedir, 'IRC_*'))
        splitnames = [os.path.split(name) for name in names]
        
        p = re.compile('IRC_\d{4,4}')
        self.subjects = [int(name[4:8]) for base, name in splitnames 
                         if not (p.match(name[-8:]) is None)]
        if samplerate is not None:
            raise ValueError('Custom samplerate not supported.')
        self.samplerate = samplerate

    def load_subject(self, subject, rounddot5 = False):


        subject = str(subject)
        if subject[0] == '3':
            # this is the case only for stuffed animals recordings
            # IRC_30..
            samplerate = 192*kHz
        else:
            samplerate = 44.1*kHz
        ok = False
        k = 0
        while k < len(self.basedir) and not ok:
            try:
                filename = os.path.join(self.basedir[k], 'IRC_' + subject)
                if self.compensated:
                    filename = os.path.join(filename, 'COMPENSATED/MAT/HRIR/IRC_' + subject + '_C_HRIR.mat')
                else:
                    filename = os.path.join(filename, 'RAW/MAT/HRIR/IRC_' + subject + '_R_HRIR.mat')
                m = loadmat(filename, struct_as_record=True)
                ok = True
            except IOError:
                ok = False
            k += 1
        if not ok:
            raise IOError("Couldn't find the HRTF files for subject "+str(subject))

        if 'l_hrir_S' in m.keys(): # RAW DATA
            affix = '_hrir_S'
        else:                      # COMPENSATED DATA
            affix = '_eq_hrir_S'
        l, r = m['l' + affix], m['r' + affix]

        azim = l['azim_v'][0][0][:, 0]
        elev = l['elev_v'][0][0][:, 0]
        if len(azim) == len(elev) and len(azim) == 1:
            # it is the case with IRCAM_New db
            # - the coordinates are 1xN instead of Nx1
            # - some measures that should be at the same elevation are
            # at very close but different elevations (7.47
            # vs. 7.5). This is annoying for interpolation. Hence I
            # allow one to round the elevations
            conv = lambda x : x
            if rounddot5:
                conv = lambda x: np.round(2*x)/2
            azim = conv(l['azim_v'][0][0][0, :])
            elev = l['elev_v'][0][0][0, :]
        coords = make_coordinates(azim=azim, elev=elev)
        l = l['content_m'][0][0]
        r = r['content_m'][0][0]
        # self.data has shape (num_ears=2, num_indices, hrir_length)
        data = vstack((reshape(l, (1,) + l.shape), reshape(r, (1,) + r.shape)))
        hrtfset = HRTFSet(data, samplerate, coords)
        hrtfset.name = 'IRCAM_'+subject
        return hrtfset
