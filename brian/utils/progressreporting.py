import sys, time

__all__ = ['ProgressReporter']

def time_rep(t):
    '''
    Textual representation of time t given in seconds
    '''
    t = int(t)
    if t < 60:
        return str(t) + 's'
    secs = t % 60
    mins = t // 60
    if mins < 60:
        return str(mins) + 'm ' + str(secs) + 's'
    mins = mins % 60
    hours = t // (60 * 60)
    if hours < 24:
        return str(hours) + 'h ' + str(mins) + 'm ' + str(secs) + 's'
    hours = hours % 24
    days = t // (60 * 60 * 24)
    return str(days) + 'd ' + str(hours) + 'h ' + str(mins) + 'm ' + str(secs) + 's'

def make_text_report(elapsed, complete):
    s = str(int(100 * complete)) + '% complete, '
    s += time_rep(elapsed) + ' elapsed'
    if complete > .001:
        remtime = elapsed / complete - elapsed
        s += ', approximately ' + time_rep(remtime) + ' remaining.'
    else:
        s += '.'
    return s

def build_text_reporter(output_stream):
    def text_report(elapsed, complete):
        s = make_text_report(elapsed, complete) + '\n'
        output_stream.write(s)
        output_stream.flush()
    return text_report


class ProgressReporter(object):
    '''
    Standard text and graphical progress reports
    
    Initialised with arguments:
    
    ``report``
        
        Can be one of the following strings:
    
        ``'print'``, ``'text'``, ``'stdout'``
            Reports progress to standard console.
        ``'stderr'``
            Reports progress to error console.
        ``'graphical'``, ``'tkinter'``
            A simple graphical progress bar using Tkinter.
        
        Alternatively, it can be any output stream in which
        case text reports will be sent to it, or a custom callback function
        ``report(elapsed, complete)`` taking arguments ``elapsed``
        the amount of time that has passed and ``complete`` the fraction of
        the computation finished.
    
    ``period``
        How often reports should be generated in seconds.
        
    ``first_report``
        The time of the first report (nothing will be done before this amount
        of time has elapsed).
    
    Methods:
    
    .. method:: start()
    
        Call at the beginning of a task to start timing it.
    
    .. method:: finish()
    
        Call at the end of a task to finish timing it. Note that
        with the Tkinter class, if you do not call this it will
        stop the Python script from finishing, stopping memory
        from being freed up.
    
    .. method:: update(complete)
    
        Call with the fraction of the task (or subtask if
        ``subtask()`` has been called) completed, between
        0 and 1.
        
    .. method:: subtask(complete, tasksize)
    
        After calling ``subtask(complete, tasksize)``,
        subsequent calls to update will report progress
        between a fraction ``complete`` and ``complete+tasksize``
        of the total task. ``complete`` represents the amount
        of the total task completed at the beginning of the
        task, and ``tasksize`` the size of the subtask as a
        proportion of the whole task.
        
    .. method:: equal_subtask(tasknum, numtasks)
    
        If a task can be divided into ``numtasks`` equally
        sized subtasks, you can use this method instead of
        ``subtask``, where ``tasknum`` is the number of
        the subtask about to start.
    '''
    def __init__(self, report, period=10.0, first_report=-1.0):
        self.period = float(period)
        #self.report = get_reporter(report)
        self.report = None
        self.report_option = report
        self.first_report = first_report
        self.first_time = 0
        self.start() # just in case the user forgets to call start()

    def start(self):
        self.start_time = time.time()
        self.next_report_time = self.start_time + self.period
        self.first_time = self.start_time + self.first_report
        self.subtask_complete = 0.0
        self.subtask_size = 1.0

    def finish(self):
        self.update(1)

    def subtask(self, complete, tasksize):
        self.subtask_complete = complete
        self.subtask_size = tasksize

    def equal_subtask(self, tasknum, numtasks):
        self.subtask(float(tasknum) / float(numtasks), 1. / numtasks)

    def update(self, complete):
        cur_time = time.time()
        totalcomplete = self.subtask_complete + complete * self.subtask_size
        if cur_time > self.first_time:
            if self.report is None:
                self.report = get_reporter(self.report_option)
            if cur_time > self.next_report_time or totalcomplete == 1.0 or totalcomplete == 1:
                self.next_report_time = cur_time + self.period
                elapsed = time.time() - self.start_time
                self.report(elapsed, totalcomplete)


def get_reporter(report):
    if report == 'print' or report == 'text' or report == 'stdout':
        report = build_text_reporter(sys.stdout)
    elif report == 'stderr':
        report = build_text_reporter(sys.stderr)
    elif hasattr(report, 'write') and hasattr(report, 'flush'):
        report = build_text_reporter(report)
    elif report == 'graphical' or report == 'tkinter':
        import Tkinter
        class ProgressBar(object):
            '''
            Adapted from: http://code.activestate.com/recipes/492230/
            '''
            # Create Progress Bar
            def __init__(self, width, height):
                self.__root = Tkinter.Tk()
                self.__root.resizable(False, False)
                self.__root.title('Progress Bar')
                self.__canvas = Tkinter.Canvas(self.__root, width=width, height=height)
                self.__canvas.grid()
                self.__width = width
                self.__height = height
            # Open Progress Bar
            def open(self):
                self.__root.deiconify()
            # Close Progress Bar
            def close(self):
                self.__root.withdraw()
            # Update Progress Bar
            def update(self, ratio, newtitle=None):
                self.__canvas.delete(Tkinter.ALL)
                self.__canvas.create_rectangle(0, 0, self.__width * ratio, \
                                               self.__height, fill='blue')
                if newtitle is not None:
                    self.__root.title(newtitle)
                self.__root.update()
        pb = ProgressBar(500, 20)
        pb.closed = False
        def report(elapsed, complete):
            try:
                if complete == 1.0 and pb.closed == False:
                    pb.close()
                    pb.closed = True
                else:
                    pb.update(complete, make_text_report(elapsed, complete))
            except Tkinter.TclError:
                # exception handling in the case that the user shuts the window
                pass
    return report

if __name__ == '__main__':
    import time
    report = ProgressReporter('stderr', 0.1, 2.0)
    report.start()
    for i in xrange(30):
        print 'iteration', i
        time.sleep(0.1)
        report.update(i/30.0)
    report.finish()
    