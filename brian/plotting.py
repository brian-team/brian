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

__all__=['plot','show','figure','xlabel','ylabel','title','axis','raster_plot','hist_plot']

try:
    from pylab import plot,show,figure,xlabel,ylabel,title,axis
    import pylab
except:
    plot, show, figure, xlabel, ylabel, title, axis = (None,)*7
from stdunits import *
import magic
from connection import *
from monitor import *
from monitor import HistogramMonitorBase
import warnings
from log import *

def _take_options(myopts,givenopts):
    """Takes options from one dict into another
    
    Any key defined in myopts and givenopts will be removed
    from givenopts and placed in myopts.
    """
    for k in myopts.keys():
        if k in givenopts:
            myopts[k] = givenopts.pop(k)


def raster_plot(*monitors,**plotoptions):
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
    ``newfigure=True``
        set to ``False`` not to create a new figure with pylab's ``figure()`` function
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
    """
    if len(monitors)==0:
        (monitors,monitornames) = magic.find_instances(SpikeMonitor)
    if len(monitors):
        # OPTIONS
        # Defaults
        myopts = {"title":"", "xlabel":"Time (ms)", "showplot":False, "showgrouplines":False,\
                  "spacebetweengroups":0.0, "grouplinecol":"k", 'newfigure':False}
        if len(monitors)==1:
            myopts["ylabel"]='Neuron number'
        else:
            myopts["ylabel"]='Group number'
        # User options
        _take_options(myopts,plotoptions)
        # PLOTTING ROUTINE
        st = []
        sn = []
        spacebetween = myopts['spacebetweengroups']
        for (m,i) in zip(monitors,range(len(monitors))):
            st = st + [float(a[1]/ms) for a in m.spikes]
            if len(monitors)==1:
                sn = sn + [a[0] for a in m.spikes]
            else:
                sn = sn + [float(i)+(1-spacebetween)*float(a[0])/float(len(m.source)) for a in m.spikes]
        if myopts['newfigure']:
            pylab.figure()
        pylab.plot(st,sn,'.',**plotoptions)
        if myopts['showgrouplines']:
            for i in range(len(monitors)):
                pylab.axhline(i,color=myopts['grouplinecol'])
                pylab.axhline(i+(1-spacebetween),color=myopts['grouplinecol'])
        pylab.ylabel(myopts['ylabel'])
        pylab.xlabel(myopts['xlabel'])
        pylab.title(myopts["title"])
        if myopts["showplot"]:
            pylab.show()


def hist_plot(histmon=None,**plotoptions):
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
        if len(histmons)==0:
            raise TypeError, "No histogram monitors found."
        elif len(histmons)>1:
            log_info('brian.hist_plot', "Found more than one histogram monitor, using first one.")
        histmon = histmons[0]
    # OPTIONS
    # Defaults
    myopts = {"title":"", "xlabel":"Time (ms)", "ylabel":"Count", "showplot":False,'newfigure':True }
    # User options
    _take_options(myopts,plotoptions)
    # PLOTTING ROUTINE
    if myopts['newfigure']:
        pylab.figure()
    pylab.bar(histmon.bins[:-1]/ms,histmon.count[:-1],(histmon.bins[1:]-histmon.bins[:-1])/ms,**plotoptions)
    pylab.ylabel(myopts['ylabel'])
    pylab.xlabel(myopts['xlabel'])
    pylab.title(myopts["title"])
    if myopts["showplot"]:
        pylab.show()
        