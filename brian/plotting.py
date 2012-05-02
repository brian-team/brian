# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
"""
Plotting routines for Brian

Functions:

* ``raster_plot(monitors...,options...)``
* ``hist_plot(monitor,options...)``
"""

__docformat__ = "restructuredtext en"

__all__ = ['plot', 'show', 'figure', 'xlabel', 'ylabel', 'title', 'axis',
           'raster_plot', 'raster_plot_spiketimes', 'hist_plot']

try:
    from pylab import plot, show, figure, xlabel, ylabel, title, axis, xlim
    import pylab, matplotlib
except:
    plot, show, figure, xlabel, ylabel, title, axis, xlim = (None,)*8
from stdunits import *
import magic
from connections import *
from monitor import *
from monitor import HistogramMonitorBase
from network import network_operation
from clock import EventClock
import warnings
from log import *
from numpy import amax, amin, array, hstack
import bisect

def _take_options(myopts, givenopts):
    """Takes options from one dict into another
    
    Any key defined in myopts and givenopts will be removed
    from givenopts and placed in myopts.
    """
    for k in myopts.keys():
        if k in givenopts:
            myopts[k] = givenopts.pop(k)


def raster_plot(*monitors, **additionalplotoptions):
    """Raster plot of a :class:`SpikeMonitor`
    
    **Usage**
    
    ``raster_plot(monitor,options...)``
        Plots the spike times of the monitor
        on the x-axis, and the neuron number on the y-axis
    ``raster_plot(monitor0,monitor1,...,options...)``
        Plots the spike times
        for all the monitors given, with y-axis defined by placing a spike
        from neuron n of m in monitor i at position i+n/m
    ``raster_plot(options...)``
        Guesses the monitors to plot automagically
    
    **Options**
    
    Any of PyLab options for the ``plot`` command can be given, as well as:
    
    ``showplot=False``
        set to ``True`` to run pylab's ``show()`` function
    ``newfigure=False``
        set to ``True`` to create a new figure with pylab's ``figure()`` function
    ``xlabel``
        label for the x-axis
    ``ylabel``
        label for the y-axis
    ``title``
        title for the plot    
    ``showgrouplines=False``
        set to ``True`` to show a line between each monitor
    ``grouplinecol``
        colour for group lines
    ``spacebetweengroups``
        value between 0 and 1 to insert a space between
        each group on the y-axis
    ``refresh``
        Specify how often (in simulation time) you would like the plot to
        refresh. Note that this will only work if pylab is in interactive mode,
        to ensure this call the pylab ``ion()`` command.
    ``showlast``
        If you are using the ``refresh`` option above, plots are much quicker
        if you specify a fixed time window to display (e.g. the last 100ms).
    ``redraw``
        If you are using more than one realtime monitor, only one of them needs
        to issue a redraw command, therefore set this to ``False`` for all but
        one of them.

    Note that with some IDEs, interactive plotting will not work with the
    default matplotlib backend, try doing something like this at the
    beginning of your script (before importing brian)::
    
        import matplotlib
        matplotlib.use('WXAgg')
        
    You may need to experiment, try WXAgg, GTKAgg, QTAgg, TkAgg.
    """
    if len(monitors) == 0:
        (monitors, monitornames) = magic.find_instances(SpikeMonitor)
    if len(monitors):
        # OPTIONS
        # Defaults
        myopts = {"title":"", "xlabel":"Time (ms)", "showplot":False,
                  "showgrouplines":False, "spacebetweengroups":0.0,
                  "grouplinecol":"k", 'newfigure':False, 'refresh':None,
                  'showlast':None, 'redraw':True}
        plotoptions = {'mew':0}
        if len(monitors) == 1:
            myopts["ylabel"] = 'Neuron number'
        else:
            myopts["ylabel"] = 'Group number'
        # User options
        _take_options(myopts, additionalplotoptions)
        plotoptions.update(additionalplotoptions)
        # PLOTTING ROUTINE
        spacebetween = myopts['spacebetweengroups']
        class SecondTupleArray(object):
            def __init__(self, obj):
                self.obj = obj
            def __getitem__(self, i):
                return float(self.obj[i][1])
            def __len__(self):
                return len(self.obj)
        def get_plot_coords(tmin=None, tmax=None):
            allsn = []
            allst = []
            for i, m in enumerate(monitors):
                mspikes = m.spikes
                if tmin is not None and tmax is not None:
                    x = SecondTupleArray(mspikes)
                    imin = bisect.bisect_left(x, tmin)
                    imax = bisect.bisect_right(x, tmax)
                    mspikes = mspikes[imin:imax]
                if len(mspikes):
                    sn, st = array(mspikes).T
                else:
                    sn, st = array([]), array([])
                st /= ms
                if len(monitors) == 1:
                    allsn = [sn]
                else:
                    allsn.append(i + ((1. - spacebetween) / float(len(m.source))) * sn)
                allst.append(st)
            sn = hstack(allsn)
            st = hstack(allst)
            if len(monitors) == 1:
                nmax = len(monitors[0].source)
            else:
                nmax = len(monitors)
            return st, sn, nmax
        st, sn, nmax = get_plot_coords()
        if myopts['newfigure']:
            pylab.figure()
        if myopts['refresh'] is None:
            line, = pylab.plot(st, sn, '.', **plotoptions)
        else:
            line, = pylab.plot([], [], '.', **plotoptions)
        if myopts['refresh'] is not None:
            pylab.axis(ymin=0, ymax=nmax)
            if myopts['showlast'] is not None:
                pylab.axis(xmin= -myopts['showlast'] / ms, xmax=0)
        ax = pylab.gca()
        if myopts['showgrouplines']:
            for i in range(len(monitors)):
                pylab.axhline(i, color=myopts['grouplinecol'])
                pylab.axhline(i + (1 - spacebetween), color=myopts['grouplinecol'])
        pylab.ylabel(myopts['ylabel'])
        pylab.xlabel(myopts['xlabel'])
        pylab.title(myopts["title"])
        if myopts["showplot"]:
            pylab.show()
        if myopts['refresh'] is not None:
            @network_operation(clock=EventClock(dt=myopts['refresh']))
            def refresh_raster_plot(clk):
                if matplotlib.is_interactive():
                    if myopts['showlast'] is None:
                        st, sn, nmax = get_plot_coords()
                        line.set_xdata(st)
                        line.set_ydata(sn)
                        ax.set_xlim(0, amax(st))
                    else:
                        st, sn, nmax = get_plot_coords(clk._t - float(myopts['showlast']), clk._t)
                        ax.set_xlim((clk.t - myopts['showlast']) / ms, clk.t / ms)
                        line.set_xdata(array(st))
                        line.set_ydata(sn)
                    if myopts['redraw']:
                        pylab.draw()
                        pylab.get_current_fig_manager().canvas.flush_events()
            monitors[0].contained_objects.append(refresh_raster_plot)


def raster_plot_spiketimes(spiketimes):
    """
    Raster plot of a list of spike times
    """
    m = Monitor()
    m.source = []
    m.spikes = spiketimes
    raster_plot(m)
    t = array(spiketimes)[:,1]


def hist_plot(histmon=None, **plotoptions):
    """Plot a histogram
    
    **Usage**
    
    ``hist_plot(histmon,options...)``
        Plot the given histogram monitor
    ``hist_plot(options...)``
        Guesses which histogram monitor to use
    
    with argument:
    
    ``histmon``
        is a monitor of histogram type
    
    **Notes**
    
    Plots only the first n-1 of n bars in the histogram, because
    the nth bar is for the interval (-,infinity).
    
    **Options**
    
    Any of PyLab options for bar can be given, as well as:
    
    ``showplot=False``
        set to ``True`` to run pylab's ``show()`` function
    ``newfigure=True``
        set to ``False`` not to create a new figure with pylab's ``figure()`` function
    ``xlabel``
        label for the x-axis
    ``ylabel``
        label for the y-axis
    ``title``
        title for the plot
    """
    if histmon is None:
        (histmons, histmonnames) = magic.find_instances(HistogramMonitorBase)
        if len(histmons) == 0:
            raise TypeError, "No histogram monitors found."
        elif len(histmons) > 1:
            log_info('brian.hist_plot', "Found more than one histogram monitor, using first one.")
        histmon = histmons[0]
    # OPTIONS
    # Defaults
    myopts = {"title":"", "xlabel":"Time (ms)", "ylabel":"Count", "showplot":False, 'newfigure':True }
    # User options
    _take_options(myopts, plotoptions)
    # PLOTTING ROUTINE
    if myopts['newfigure']:
        pylab.figure()
    pylab.bar(histmon.bins[:-1] / ms, histmon.count[:-1], (histmon.bins[1:] - histmon.bins[:-1]) / ms, **plotoptions)
    pylab.ylabel(myopts['ylabel'])
    pylab.xlabel(myopts['xlabel'])
    pylab.title(myopts["title"])
    if myopts["showplot"]:
        pylab.show()
