'''
Remote control of a Brian process, for example using an IPython shell

The process running the simulation calls something like:

server = remote_control_server()

and the IPython shell calls:

client = RemoteControlClient()

The shell can now execute and evaluate in the server process via:

spikes = client.evaluate('M.spikes')
i, t = zip(*spikes)
plot(t, i, '.')
client.execute('stop()')
'''

from brian import *
from multiprocessing.connection import Listener, Client
import select
import inspect

__all__ = ['remote_control_server', 'RemoteControlClient']

@magic_return
def remote_control_server(server=None, authkey='brian', clock=None,
                          global_ns=None, local_ns=None, level=1):
    '''
    Allows remote control (via IP) of a running Brian script
    
    Initialisation arguments:
    
    ``server``
        The IP server details, a pair (host, port). If you want to allow control
        only on the one machine (for example, by an IPython shell), leave this
        as ``None`` (which defaults to host='localhost', port=2719).
    ``authkey``
        The authentication key to allow access, change it from 'brian' if you
        are allowing access from outside (otherwise you allow others to run
        arbitrary code on your machine).
    ``clock``
        The clock specifying how often to poll for incoming commands.
    ``global_ns``, ``local_ns``, ``level``
        Namespaces in which incoming commands will be executed or evaluated,
        if you leave them blank it will be the local and global namespace of
        the frame from which this function was called (if level=1, or from
        a higher level if you specify a different level here).
    
    Once this object has been created, use a :class:`RemoteControlClient` to
    issue commands.
    
    **Example usage**
    
    Main simulation code includes a line like this::
    
        server = remote_control_server()
        
    In an IPython shell you can do something like this::
    
        spikes = client.evaluate('M.spikes')
        i, t = zip(*spikes)
        plot(t, i, '.')
        client.execute('stop()')
    '''
    if server is None:
        server = ('localhost', 2719)
    frame = inspect.stack()[level+1][0]
    ns_global, ns_local=frame.f_globals,frame.f_locals
    if global_ns is None:
        global_ns = frame.f_globals        
    if local_ns is None:
        local_ns = frame.f_locals
    listener = Listener(server, authkey=authkey)
    connholder = [None]
    @network_operation(clock=clock)
    def server_check():
        conn = connholder[0]
        if conn is None:
            # This is kind of a hack. The multiprocessing.Listener class doesn't
            # allow you to tell if an incoming connection has been requested
            # without accepting that connection, which means if nothing attempts
            # to connect it will wait forever for something to connect. What
            # we do here is check if there is any incoming data on the
            # underlying IP socket used internally by multiprocessing.Listener.
            socket = listener._listener._socket
            sel, _, _ = select.select([socket], [], [], 0)
            if len(sel):
                conn = listener.accept()
                connholder[0] = conn
        if conn is None:
            return
        if not conn.poll():
            return
        job = conn.recv()
        jobtype, jobargs = job
        try:
            if jobtype=='exec':
                exec jobargs in global_ns, local_ns
                result = None
            elif jobtype=='eval':
                result = eval(jobargs, global_ns, local_ns)
        except Exception, e:
            # if it raised an exception, we return that exception and the
            # client can then raise it.
            result = e
        conn.send(result)
    return server_check

class RemoteControlClient(object):
    '''
    Used to remotely control (via IP) a running Brian script
    
    Initialisation arguments:
    
    ``server``
        The IP server details, a pair (host, port). If you want to allow control
        only on the one machine (for example, by an IPython shell), leave this
        as ``None`` (which defaults to host='localhost', port=2719).
    ``authkey``
        The authentication key to allow access, change it from 'brian' if you
        are allowing access from outside (otherwise you allow others to run
        arbitrary code on your machine).
    
    Has the following methods:
    
    .. method:: execute(code)
    
        Executes the specified code in the server process.
        If it raises an
        exception, the server process will catch it and reraise it in the
        client process.
        
    .. method:: evaluate(code)
    
        Evaluate the code in the server process and return the result.
        If it raises an
        exception, the server process will catch it and reraise it in the
        client process.
 
    **Example usage**
    
    Main simulation code includes a line like this::
    
        server = remote_control_server()
        
    In an IPython shell you can do something like this::
    
        spikes = client.evaluate('M.spikes')
        i, t = zip(*spikes)
        plot(t, i, '.')
        client.execute('stop()')
    '''
    def __init__(self, server=None, authkey='brian'):
        if server is None:
            server = ('localhost', 2719)
        self.client = Client(server, authkey=authkey)
    def execute(self, code):
        self.client.send(('exec', code))
        result = self.client.recv()
        if isinstance(result, Exception):
            raise result
    def evaluate(self, code):
        self.client.send(('eval', code))
        result = self.client.recv()
        if isinstance(result, Exception):
            raise result
        return result
 