from datamanager import DataManager
import multiprocessing
from Queue import Empty as QueueEmpty
import Tkinter
from brian.utils.progressreporting import make_text_report
import inspect
import time
import os
from numpy import ndarray, zeros

__all__ = ['run_tasks']

# This is the default task class used if the user provides only a function
class FunctionTask(object):
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwds):
        # If the function has a 'report' argument, we pass it the reporter
        # function that will have been passed in kwds (see task_compute)
        fc = self.func.func_code
        if 'report' in fc.co_varnames[:fc.co_argcount] or fc.co_flags&8:
            return self.func(*args, **kwds)
        else:
            return self.func(*args)

def run_tasks(dataman, task, items, gui=True, poolsize=0,
              initargs=None, initkwds=None, verbose=None,
              numitems=None):
    '''
    Run a series of tasks using multiple CPUs on a single computer.
    
    Initialised with arguments:
    
    ``dataman``
        The :class:`~brian.tools.datamanager.DataManager` object used to store
        the results in, see below.
    ``task``
        The task function or class (see below).
    ``items``
        A sequence (e.g. list or iterator) of arguments to be passed to the
        task.
    ``gui=True``
        Whether or not to use a Tkinter based GUI to show progress and terminate
        the task run.
    ``poolsize=0``
        The number of CPUs to use. If the value is 0, use all available CPUs,
        if it is -1 use all but one CPU, etc.
    ``initargs``, ``initkwds``
        If ``task`` is a class, these are the initialisation arguments and
        keywords for the class.
    ``verbose=None``
        Specify True or False to print out every progress message (defaults to
        False if the GUI is used, or True if not).
    ``numitems=None``
        For iterables (rather than fixed length sequences), if you specify the
        number of items, an estimate of the time remaining will be given.
        
    The task (defined by a function or class, see below) will be called on each
    item in ``items``, and the results saved to ``dataman``. Results are stored
    in the format ``(key, val)`` where ``key`` is a unique but meaningless
    identifier. Results can be retrieved using ``dataman.values()`` or (for
    large data sets that should be iterated over) ``dataman.itervalues()``.
    
    The task can either be a function or a class. If it is a function, it will
    be called for each item in ``items``. If the items are tuples, the function
    will be called with those tuples as arguments (e.g. if the item is
    ``(1,2,3)`` the function will be called as ``task(1, 2, 3)``). If the task
    is a class, it can have an ``__init__`` method that is called once for
    each process (each CPU) at the beginning of the task run. If the ``__init__``
    method has a ``process_number`` argument, it will be passed an integer value
    from 0 to ``numprocesses-1`` giving the number of the process (note, this is
    not the process ID). The class should
    define a ``__call__`` method that behaves the same as above for ``task``
    being a function. In both cases (function or class), if the arguments
    include a keyword ``report`` then it will be passed a value that can be
    passed as the ``report`` keyword in Brian's :func:`run` function to give
    feedback on the simulation as it runs. A ``task`` function can also set
    ``self.taskname`` as a string that will be displayed on the GUI to give
    additional information.
    
    .. warning::
        On Windows, make sure that :func:`run_tasks` is only called from
        within a block such as::
        
            if __name__=='__main__':
                run_tasks(...)
                
        Otherwise, the program will go into a recursive loop.
        
    Note that this class only allows you to run tasks on a single computer, to
    distribute work over multiple computers, we suggest using
    `Playdoh <http://code.google.com/p/playdoh/>`__.
    '''
    # User can provide task as a class or a function, if its a function we
    # we use the default FunctionTask
    if not inspect.isclass(task):
        f = task
        initargs = (task,)
        task = FunctionTask
    else:
        f = task.__call__
    fc = f.func_code
    if 'report' in fc.co_varnames[:fc.co_argcount] or fc.co_flags&8:
        will_report = True
    else:
        will_report = False
    if numitems is None and isinstance(items, (list, tuple, ndarray)):
        numitems = len(items)
    # This will be used to provide process safe access to the data manager
    # (so that multiple processes do not attempt to write to the session at
    # the same time)
    session = dataman.locking_computer_session()
    if poolsize<=0:
        numprocesses = poolsize+multiprocessing.cpu_count()
    elif poolsize>0:
        numprocesses = poolsize
    manager = multiprocessing.Manager()
    # We have to send the process number to the initializer via this silly
    # queue because of a limitation of multiprocessing
    process_number_queue = manager.Queue()
    for n in range(numprocesses):
        process_number_queue.put(n)
    # This will be used to send messages about the status of the run, i.e.
    # percentage complete
    message_queue = manager.Queue()
    if initargs is None:
        initargs = ()
    if initkwds is None:
        initkwds = {}
    pool = multiprocessing.Pool(processes=numprocesses,
                                initializer=pool_initializer,
                                initargs=(process_number_queue, message_queue,
                                          dataman, session,
                                          task, initargs, initkwds))
    results = pool.imap_unordered(task_compute, items)
    # We use this to map process IDs to task number, so that we can show the
    # information on the GUI in a consistent fashion
    pid_to_id = dict((pid, i) for i, pid in enumerate([p.pid for p in pool._pool]))
    start = time.time()
    stoprunningsim = [False]
    def terminate_sim():
        # We acquire the datamanager session lock so that if a process is in the
        # middle of writing data, it won't be terminated until its finished,
        # meaning we can safely terminate the process without worrying about
        # data loss.
        session.acquire()
        pool.terminate()
        session.release()
        stoprunningsim[0] = True
    if gui:
        if verbose is None:
            verbose = False
        controller = GuiTaskController(numprocesses, terminate_sim,
                                       verbose=verbose, will_report=will_report)
    else:
        if verbose is None:
            verbose = True
        controller = TextTaskController(numprocesses, terminate_sim, verbose=verbose)
    for i in range(numprocesses):
        controller.update_process(i, 0, 0, 'No task information')
    i = 0
    controller.update_overall(0, numitems)
    def empty_message_queue():
        while not message_queue.empty():
            try:
                pid, taskname, elapsed, complete = message_queue.get_nowait()
                controller.update_process(pid_to_id[pid], elapsed, complete, taskname)
            except QueueEmpty:
                break
        controller.update()
        
    while True:
        try:
            # This waits 0.1s for a new result, and otherwise raises a
            # TimeoutError that allows the GUI to update the percentage
            # complete
            nextresult = results.next(0.1)
            empty_message_queue()
            i = i+1
            elapsed = time.time()-start
            complete = 0.0
            controller.update_overall(i, numitems)
        except StopIteration:
            terminate_sim()
            print 'Finished.'
            break
        except (KeyboardInterrupt, SystemExit):
            terminate_sim()
            print 'Terminated task processes'
            raise
        except multiprocessing.TimeoutError:
            empty_message_queue()
            if stoprunningsim[0]:
                print 'Terminated task processes'
                break
    controller.destroy()


# We store these values as global values, which are initialised by
# pool_initializer on each process
task_object = None
task_dataman = None
task_session = None
task_message_queue = None

def pool_initializer(process_number_queue, message_queue, dataman, session,
                     task, initargs, initkwds):
    global task_object, task_dataman, task_session, task_message_queue
    n = process_number_queue.get()
    init_method = task.__init__
    fc = init_method.func_code
    # Checks if there is a process_number argument explicitly given in the
    # __init__ method of the task class, the co_flags&8 checks i there is a
    # **kwds parameter in the definition
    if 'process_number' in fc.co_varnames[:fc.co_argcount] or fc.co_flags&8:
        initkwds['process_number'] = n
    task_object = task(*initargs, **initkwds)
    task_dataman = dataman
    task_session = session
    task_message_queue = message_queue

def task_reporter(elapsed, complete):
    # If the task class defines a task name, we can display it with the
    # percentage complete
    if hasattr(task_object, 'taskname'):
        taskname = task_object.taskname
    else:
        taskname = 'No task information'
    # This queue is used by the main loop in run_tasks
    task_message_queue.put((os.getpid(), taskname, elapsed, complete))

def task_compute(args):
    if not isinstance(args, tuple):
        args = (args,)
    # We check if the task function has a report argument, and if it does we
    # pass it task_reporter so that it can integrate with the GUI
    kwds = {}
    fc = task_object.__call__.func_code
    if 'report' in fc.co_varnames[:fc.co_argcount] or fc.co_flags&8:
        kwds['report'] = task_reporter
    result = task_object(*args, **kwds)
    # Save the results, with a unique key, to the locking session of the dataman
    task_session[task_dataman.make_unique_key()] = result

class TaskController(object):
    def __init__(self, processes, terminator, verbose=True):
        self.verbose = verbose
        self.completion = zeros(processes)
        self.numitems, self.numdone = None, 0
        self.start_time = time.time()
    def update_process(self, i, elapsed, complete, msg):
        self.completion[i] = complete%1.0
        if self.verbose:
            print 'Process '+str(i)+': '+make_text_report(elapsed, complete)+': '+msg
            _, msg = self.get_overall_completion()
            print msg
    def get_overall_completion(self):
        complete = 0.0
        numitems, numdone = self.numitems, self.numdone
        elapsed = time.time()-self.start_time
        if numitems is not None:
            complete = (numdone+sum(self.completion))/numitems
        txt = 'Overall, '+str(numdone)+' done'
        if numitems is not None:
            txt += ' of '+str(numitems)+': '+make_text_report(elapsed, complete)
        return complete, txt
    def update_overall(self, numdone, numitems):
        self.numdone = numdone
        self.numitems = numitems
    def recompute_overall(self):
        pass
    def update(self):
        pass
    def destroy(self):
        pass

class TextTaskController(TaskController):
    def update_overall(self, numdone, numitems):
        TaskController.update_overall(self, numdone, numitems)
        _, msg = self.get_overall_completion()
        print msg

# task control GUI
class GuiTaskController(Tkinter.Tk, TaskController):
    def __init__(self, processes, terminator, width=600, verbose=False,
                 will_report=True):
        Tkinter.Tk.__init__(self, None)
        TaskController.__init__(self, processes, terminator, verbose=verbose)
        self.parent = None
        self.grid()
        button = Tkinter.Button(self, text='Terminate task',
                                command=terminator)
        button.grid(column=0, row=0)
        self.pb_width = width
        self.progressbars = []
        numbars = 1
        self.will_report = will_report
        if will_report:
            numbars += processes
        for i in xrange(numbars):
            can = Tkinter.Canvas(self, width=width, height=30)
            can.grid(column=0, row=1+i)
            can.create_rectangle(0, 0, width, 30, fill='#aaaaaa')
            if i<numbars-1:
                col = '#ffaaaa'
            else:
                col = '#aaaaff'
            r = can.create_rectangle(0, 0, 0, 30, fill=col, width=0)
            t = can.create_text(width/2, 15, text='')
            self.progressbars.append((can, r, t))
        self.title('Task control')
        
    def update_process(self, i, elapsed, complete, msg):
        TaskController.update_process(self, i, elapsed, complete, msg)
        if self.will_report:
            can, r, t = self.progressbars[i]
            can.itemconfigure(t, text='Process '+str(i)+': '+make_text_report(elapsed, complete)+': '+msg)
            can.coords(r, 0, 0, int(self.pb_width*complete), 30)
        self.recompute_overall()
        
    def update_overall(self, numdone, numitems):
        TaskController.update_overall(self, numdone, numitems)
        self.recompute_overall()
    
    def recompute_overall(self):
        complete, msg = TaskController.get_overall_completion(self)
        numitems = self.numitems
        can, r, t = self.progressbars[-1]
        can.itemconfigure(t, text=msg)
        if numitems is not None:
            can.coords(r, 0, 0, int(self.pb_width*complete), 30)
        self.update()
