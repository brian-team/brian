#!/usr/bin/env python
'''
A complicated example of using multiprocessing for multiple runs of a simulation
with different parameters, using a GUI to monitor and control the runs.

This example features:

* An indefinite number of runs, with a set of parameters for each run generated
  at random for each run.
* A plot of the output of all the runs updated as soon as each run is completed.
* A GUI showing how long each process has been running for and how long until
  it completes, and with a button allowing you to terminate the runs.

A simpler example is in ``examples/multiprocessing/multiple_runs_simple.py``.
'''

# We use Tk as the backend for the GUI and matplotlib so as to avoid any
# threading conflicts
import matplotlib
matplotlib.use('TkAgg')

from brian import *
import Tkinter, time, multiprocessing, os
from brian.utils.progressreporting import make_text_report
from Queue import Empty as QueueEmpty


class SimulationController(Tkinter.Tk):
    '''
    GUI, uses Tkinter and features a progress bar for each process, and a callback
    function for when the terminate button is clicked.
    '''
    def __init__(self, processes, terminator, width=600):
        Tkinter.Tk.__init__(self, None)
        self.parent = None
        self.grid()
        button = Tkinter.Button(self, text='Terminate simulation',
                                command=terminator)
        button.grid(column=0, row=0)
        self.pb_width = width
        self.progressbars = []
        for i in xrange(processes):
            can = Tkinter.Canvas(self, width=width, height=30)
            can.grid(column=0, row=1 + i)
            can.create_rectangle(0, 0, width, 30, fill='#aaaaaa')
            r = can.create_rectangle(0, 0, 0, 30, fill='#ffaaaa', width=0)
            t = can.create_text(width / 2, 15, text='')
            self.progressbars.append((can, r, t))
        self.results_text = Tkinter.Label(self, text='Computed 0 results, time taken: 0s')
        self.results_text.grid(column=0, row=processes + 1)
        self.title('Simulation control')

    def update_results(self, elapsed, complete):
        '''
        Method to update the total number of results computed and the amount of time taken.
        '''
        self.results_text.config(text='Computed ' + str(complete) + ', time taken: ' + str(int(elapsed)) + 's')
        self.update()

    def update_process(self, i, elapsed, complete, msg):
        '''
        Method to update the status of a given process.
        '''
        can, r, t = self.progressbars[i]
        can.itemconfigure(t, text='Process ' + str(i) + ': ' + make_text_report(elapsed, complete) + ': ' + msg)
        can.coords(r, 0, 0, int(self.pb_width * complete), 30)
        self.update()

def sim_mainloop(pool, results, message_queue):
    '''
    Monitors results of a simulation as they arrive
    
    pool is the multiprocessing.Pool that the processes are running in,
    results is the AsyncResult object returned by Pool.imap_unordered which
    returns simulation results asynchronously as and when they are ready,
    and message_queue is a multiprocessing.Queue used to communicate between
    child processes and the server process. In this case, we use this Queue to
    send messages about the percent complete and time elapsed for each run.
    '''
    # We use this to enumerate the processes, mapping their process IDs to an int
    # in the range 0:num_processes.
    pid_to_id = dict((pid, i) for i, pid in enumerate([p.pid for p in pool._pool]))
    num_processes = len(pid_to_id)
    start = time.time()
    stoprunningsim = [False]
    # This function terminates all the pool's child processes, it is used as
    # the callback function called when the terminate button on the GUI is clicked.
    def terminate_sim():
        pool.terminate()
        stoprunningsim[0] = True
    controller = SimulationController(num_processes, terminate_sim)
    for i in range(num_processes):
        controller.update_process(i, 0, 0, 'no info yet')
    i = 0
    while True:
        try:
            # If there is a new result (the 0.1 means wait 0.1 seconds for a
            # result before giving up) then this try clause will execute, otherwise
            # a TimeoutError will occur and the except clause afterwards will
            # execute.
            weight, numspikes = results.next(0.1)
            # if we reach here, we have a result to plot, so we plot it and
            # update the GUI
            plot_result(weight, numspikes)
            i = i + 1
            controller.update_results(time.time() - start, i)
        except multiprocessing.TimeoutError:
            # if we're still waiting for a new result, we can process events in
            # the message_queue and update the GUI if there are any.
            while not message_queue.empty():
                try:
                    # messages here are of the form: (pid, elapsed, complete)
                    # where pid is the process ID of the child process, elapsed
                    # is the amount of time elapsed, and complete is the
                    # fraction of the run completed. See function how_many_spikes
                    # to see where these messages come from.
                    pid, elapsed, complete = message_queue.get_nowait()
                    controller.update_process(pid_to_id[pid], elapsed, complete, '')
                except QueueEmpty:
                    break
            controller.update()
            if stoprunningsim[0]:
                print 'Terminated simulation processes'
                break
    controller.destroy()

def plot_result(weight, numspikes):
    plot([weight], [numspikes], '.', color=(0, 0, 0.5))
    axis('tight')
    draw() # this forces matplotlib to redraw

# Note that how_many_spikes only takes one argument, which is a tuple of
# its actual arguments. The reason for this is that Pool.imap_unordered can only
# pass a single argument to the function its applied to, but that argument can
# be a tuple...
def how_many_spikes((excitatory_weight, message_queue)):
    reinit_default_clock()
    clear(True)

    eqs = '''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''
    P = NeuronGroup(4000, eqs, threshold= -50 * mV, reset= -60 * mV)
    P.v = -60 * mV + 10 * mV * rand(len(P))
    Pe = P.subgroup(3200)
    Pi = P.subgroup(800)
    Ce = Connection(Pe, P, 'ge')
    Ci = Connection(Pi, P, 'gi')
    Ce.connect_random(Pe, P, 0.02, weight=excitatory_weight)
    Ci.connect_random(Pi, P, 0.02, weight= -9 * mV)
    M = SpikeMonitor(P)

    # This reporter function is called every second, and it sends a message to
    # the server process updating the status of the current run.
    def reporter(elapsed, complete):
        message_queue.put((os.getpid(), elapsed, complete))

    run(4000 * ms, report=reporter, report_period=1 * second)

    return (excitatory_weight, M.nspikes)


if __name__ == '__main__':
    numprocesses = None # number of processes to use, set to None to have one per CPU
    # We have to use a Queue from the Manager to send messages from client
    # processes to the server process
    manager = multiprocessing.Manager()
    message_queue = manager.Queue()
    pool = multiprocessing.Pool(processes=numprocesses)
    # This generator function repeatedly generates random sets of parameters
    # to pass to the how_many_spikes function
    def args():
        while True:
            weight = rand()*3.5 * mV
            yield (weight, message_queue)
    # imap_unordered returns an AsyncResult object which returns results as
    # and when they are ready, we pass this results object which is returned
    # immediately to the sim_mainloop function which monitors this, updates the
    # GUI and plots the results as they come in.
    results = pool.imap_unordered(how_many_spikes, args())
    ion() # this puts matplotlib into interactive mode to plot as we go
    sim_mainloop(pool, results, message_queue)
