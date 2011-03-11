from uuid import uuid4
from getpass import getuser
import os
import shelve
import platform
from glob import glob
from multiprocessing import Manager

__all__ = ['DataManager']


class LockingSession(object):
    def __init__(self, dataman, session_filename):
        self.dataman = dataman
        self.session_filename = session_filename
        self.lock = Manager().Lock()

    def acquire(self):
        self.lock.acquire()
        self.session = DataManager.shelf(self.session_filename)

    def release(self):
        self.session.close()
        self.session = None
        self.lock.release()

    def __getitem__(self, item):
        self.acquire()
        ret = self.session[item]
        self.release()
        return ret

    def __setitem__(self, item, value):
        self.acquire()
        self.session[item] = value
        self.release()


class DataManager(object):
    '''
    DataManager is a simple class for managing data produced by multiple
    runs of a simulation carried out in separate processes or machines. Each
    process is assigned a unique ID and Python Shelf object to write its data
    to. Each shelf is a dictionary whose keys must be strings. The DataManager
    can collate information across multiple shelves using the get(key) method,
    which returns a dictionary with keys the unique session names, and values
    the value written in that session (typically only the values will be of
    interest). If each value is a tuple or list then you can use the
    get_merged(key) to get a concatenated list. If the data type is more
    complicated you can use the get(key) method and merge by hand. The idea
    is each process generates files with names that do not interfere with each
    other so that there are no file concurrency issues, and then in the data
    analysis phase, the data generated separately by each process is merged
    together.
    
    Methods:
    
    ``get(key)``
        Return dictionary with keys the session names, and values the values
        stored in that session for the given key.
    ``get_merged(key)``
        Return a single list of the merged lists or tuples if each value for
        every session is a list or tuple.
    ``get_matching(match)``
        Returns a dictionary with keys the keys matching match and values
        get(key). If match is a string, a matching key has to start with that
        string. If match is a function, a key matches if match(key).
    ``get_merged_matching(match)``
        Like get_merged(key) but across all keys that match.
    ``get_flat_matching(match)``
        Returns a straight list of every value session[key] for all sessions
        and all keys matching match.
    ``iteritems()``
        Returns all ``(key, value)`` pairs, for each Shelf file, as an iterator
        (useful for large files with too much data to be loaded into memory).
    ``itervalues()``
        Return all values, for each Shelf file, as an iterator.
    ``items()``, ``values()``
        As for ``iteritems`` and ``itervalues`` but returns a list rather than
        an iterator.
    ``itemcount()``
        Returns the total number of items across all the Shelf files.
    ``keys()``
        A list of all the keys across all sessions.
    ``session()``
        Returns a randomly named session Shelf, multiple processes can write to
        these without worrying about concurrency issues.
    ``computer_session()``
        Returns a consistently named Shelf specific to that user and computer,
        only one process can write to it without worrying about concurrency issues.
    ``locking_session()``, ``locking_computer_session()``
        Returns a LockingSession object, a limited proxy to the underlying
        Shelf which acquires and releases a lock before and after every
        operation, making it safe for concurrent access.
    ``session_filenames()``
        A list of all the shelf filenames for all sessions.
    ``make_unique_key()``
        Generates a unique key for inserting an element into a session without
        overwriting data, uses uuid4.
        
    Attributes:
    
    ``basepath``
        The base path for data files.
    ``computer_name``
        A (hopefully) unique identifier for the user and computer, consists of
        the username and the computer network name.
    ``computer_session_filename``
        The filename of the computer-specific session file. This file should
        only be accessed by one process at a time, there's no way to protect
        against concurrent write accesses causing it to be corrupted.
    '''
    def __init__(self, name, datapath=''):
        # check if directory exists, and if not, make it
        basepath = os.path.join(datapath, name + '.data')
        if not os.path.exists(basepath):
            subpaths = (name + '.data').split('/')
            curpath = ''
            for path in subpaths:
                curpath += path
                if not os.path.exists(os.path.join(datapath, curpath)):
                    os.mkdir(os.path.join(datapath, curpath))
                curpath += '/'
        self.basepath = basepath
        self.computer_name = getuser() + '.' + platform.node()
        self.computer_session_filename = self.session_filename(self.computer_name)

    def session_name(self):
        return getuser() + '.' + str(uuid4())

    def session_filename(self, session_name=None):
        if session_name is None:
            session_name = self.session_name()
        fname = os.path.normpath(os.path.join(self.basepath, session_name))
        return fname
    @staticmethod
    def shelf(fname):
        return shelve.open(fname, protocol=2)

    def session(self, session_name=None):
        return self.shelf(self.session_filename(session_name))

    def computer_session(self):
        return self.shelf(self.computer_session_filename)

    def locking_session(self, session_name=None):
        return LockingSession(self, self.session_filename(session_name))

    def locking_computer_session(self):
        return LockingSession(self, self.computer_session_filename)

    def session_filenames(self):
        return glob(os.path.join(self.basepath, '*'))

    def get(self, key):
        allfiles = self.session_filenames()
        ret = {}
        for name in allfiles:
            path, file = os.path.split(name)
            shelf = shelve.open(name, protocol=2)
            if key in shelf:
                ret[file] = shelf[key]
        return ret

    def get_merged(self, key):
        allitems = self.get(key)
        ret = []
        for _, val in allitems.iteritems():
            if isinstance(val, (list, tuple)):
                ret.extend(val)
            else:
                raise TypeError('Can only get merged items of list or tuple type, use get() method and merge by hand.')
        return ret

    def get_matching_keys(self, match):
        allkeys = self.keys()
        matching_keys = [key for key in allkeys if (callable(match) and match(key)) or key.startswith(match)]
        return matching_keys

    def get_matching(self, match):
        ret = {}
        for key in self.get_matching_keys(match):
            ret[key] = self.get(key)
        return ret

    def get_merged_matching(self, match):
        ret = []
        for key in self.get_matching_keys(match):
            ret.extend(self.get_merged(key))
        return ret

    def get_flat_matching(self, match):
        allitems = self.get_matching(match)
        ret = []
        for matching_key, matching_dict in allitems.iteritems():
            ret.extend(matching_dict.values())
        return ret

    def keys(self):
        allkeys = set([])
        for name in self.session_filenames():
            allkeys.update(set(shelve.open(name, protocol=2).keys()))
        return list(allkeys)
    
    def make_unique_key(self):
        return str(uuid4())
    
    def iteritems(self):
        allfiles = self.session_filenames()
        for name in allfiles:
            shelf = shelve.open(name, protocol=2)
            for key, value in shelf.iteritems():
                yield key, value
                
    def itervalues(self):
        for key, val in self.iteritems():
            yield val
    
    def items(self):
        return list(self.iteritems())
            
    def values(self):
        return list(self.itervalues())
    
    def itemcount(self):
        return sum(len(shelve.open(name, protocol=2)) for name in self.session_filenames())

if __name__ == '__main__':
    d = DataManager('test/testing')
    #s = d.session()
    #s['a'] = [7]
    print d.get_merged('a')

    #d = DataManager('test')
    #print d.get_merged('b')
    #print d.keys()
    #print d['b']
    #s = d.session()
    #s['b'] = 6
