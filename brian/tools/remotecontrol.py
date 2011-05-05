'''
Remote control of a Brian process, for example using an IPython shell

The process running the simulation calls something like:

server = RemoteControlServer()

and the IPython shell calls:

client = RemoteControlClient()

The shell can now execute and evaluate in the server process via:

spikes = client.evaluate('M.spikes')
i, t = zip(*spikes)
plot(t, i, '.')
client.execute('stop()')
'''

from ..network import NetworkOperation
try:
    import multiprocessing
    from multiprocessing.connection import Listener, Client
except ImportError:
    multiprocessing = None
import select
import inspect

__all__ = ['RemoteControlServer', 'RemoteControlClient']


class RemoteControlServer(NetworkOperation):
    '''
    Allows remote control (via IP) of a running Brian script
    
    Initialisation arguments:
    
    ``server``
        The IP server details, a pair (host, port). If you want to allow control
        only on the one machine (for example, by an IPython shell), leave this
        as ``None`` (which defaults to host='localhost', port=2719). To allow
        remote control, use ('', portnumber).
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
    
        server = RemoteControlServer()
        
    In an IPython shell you can do something like this::
    
        client = RemoteControlClient()
        spikes = client.evaluate('M.spikes')
        i, t = zip(*spikes)
        plot(t, i, '.')
        client.execute('stop()')
    '''
    def __init__(self, server=None, authkey='brian', clock=None,
                 global_ns=None, local_ns=None, level=0):
        if multiprocessing is None:
            raise ImportError('Cannot import the required multiprocessing module.')
        NetworkOperation.__init__(self, lambda:None, clock=clock)
        if server is None:
            server = ('localhost', 2719)
        frame = inspect.stack()[level + 1][0]
        ns_global, ns_local = frame.f_globals, frame.f_locals
        if global_ns is None:
            global_ns = frame.f_globals
        if local_ns is None:
            local_ns = frame.f_locals
        self.local_ns = local_ns
        self.global_ns = global_ns
        self.listener = Listener(server, authkey=authkey)
        self.conn = None

    def __call__(self):
        if self.conn is None:
            # This is kind of a hack. The multiprocessing.Listener class doesn't
            # allow you to tell if an incoming connection has been requested
            # without accepting that connection, which means if nothing attempts
            # to connect it will wait forever for something to connect. What
            # we do here is check if there is any incoming data on the
            # underlying IP socket used internally by multiprocessing.Listener.
            socket = self.listener._listener._socket
            sel, _, _ = select.select([socket], [], [], 0)
            if len(sel):
                self.conn = self.listener.accept()
        if self.conn is None:
            return
        conn = self.conn
        global_ns = self.global_ns
        local_ns = self.local_ns
        paused = 1
        while conn and paused != 0:
            if paused >= 0 and not conn.poll():
                return
            try:
                job = conn.recv()
            except:
                self.conn = None
                break
            jobtype, jobargs = job
            if paused == 1: paused = 0
            try:
                result = None
                if jobtype == 'exec':
                    exec jobargs in global_ns, local_ns
                elif jobtype == 'eval':
                    result = eval(jobargs, global_ns, local_ns)
                elif jobtype == 'setvar':
                    varname, varval = jobargs
                    local_ns[varname] = varval
                elif jobtype == 'pause':
                    paused = -1
                elif jobtype == 'go':
                    paused = 0
            except Exception, e:
                # if it raised an exception, we return that exception and the
                # client can then raise it.
                result = e
            conn.send(result)


class RemoteControlClient(object):
    '''
    Used to remotely control (via IP) a running Brian script
    
    Initialisation arguments:
    
    ``server``
        The IP server details, a pair (host, port). If you want to allow control
        only on the one machine (for example, by an IPython shell), leave this
        as ``None`` (which defaults to host='localhost', port=2719). To allow
        remote control, use ('', portnumber).
    ``authkey``
        The authentication key to allow access, change it from 'brian' if you
        are allowing access from outside (otherwise you allow others to run
        arbitrary code on your machine).

    Use a :class:`RemoteControlServer` on the simulation you want to control.
    
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
        
    .. method:: set(name, value)
    
        Sets the variable ``name`` (a string) to the given value (can be an
        array, etc.). Note that the variable is set in the local namespace, not
        the global one, and so this cannot be used to modify global namespace
        variables. To do that, set a local namespace variable and then
        call :meth:`~RemoteControlClient.execute` with an instruction to change
        the global namespace variable.

    .. method:: pause()
    
        Temporarily stop the simulation in the server process, continue
        simulation with the :meth:'go' method.
        
    .. method:: go()
    
        Continue a simulation that was paused.
        
    .. method:: stop()
    
        Stop a simulation, equivalent to ``execute('stop()')``.
 
    **Example usage**
    
    Main simulation code includes a line like this::
    
        server = RemoteControlServer()
        
    In an IPython shell you can do something like this::
    
        client = RemoteControlClient()
        spikes = client.evaluate('M.spikes')
        i, t = zip(*spikes)
        plot(t, i, '.')
        client.execute('stop()')
   '''
    def __init__(self, server=None, authkey='brian'):
        if multiprocessing is None:
            raise ImportError('Cannot import the required multiprocessing module.')
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
    
    def set(self, name, value):
        self.client.send(('setvar', (name, value)))
        result = self.client.recv()
        if isinstance(result, Exception):
            raise result

    def pause(self):
        self.client.send(('pause', ''))
        self.client.recv()

    def go(self):
        self.client.send(('go', ''))
        self.client.recv()

    def stop(self):
        self.execute('stop()')
