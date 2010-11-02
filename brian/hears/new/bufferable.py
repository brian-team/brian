'''
The Bufferable class serves as a base for all the other Brian.hears classes
'''

class Bufferable(object):
    '''
    Base class for Brian.hears classes
    
    Defines a buffering interface.
    '''
    def buffer_fetch(self, samples=None, duration=None):
        raise NotImplementedError
    
    def buffer_init(self):
        raise NotImplementedError
