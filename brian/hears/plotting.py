from brian import *
from erb import *

__all__ = ['log_frequency_xaxis_labels', 'log_frequency_yaxis_labels']

def log_frequency_xaxis_labels(ax=None, freqs=None):
    '''
    Sets tick positions for log-scale frequency x-axis at sensible locations.
    
    Also uses scalar representation rather than exponential (i.e. 100 rather
    than 10^2).
    
    ``ax=None``
        The axis to set, or uses ``gca()`` if ``None``.
    ``freqs=None``
        Override the default frequency locations with your preferred tick
        locations.
        
    See also: :func:`log_frequency_yaxis_labels`.
    
    Note: with log scaled axes, it can be useful to call ``axis('tight')``
    before setting the ticks.
    '''
    if ax is None:
        ax = gca()
    return log_frequency_axis_labels(ax.xaxis, freqs=freqs)

def log_frequency_yaxis_labels(ax=None, freqs=None):
    '''
    Sets tick positions for log-scale frequency x-axis at sensible locations.
    
    Also uses scalar representation rather than exponential (i.e. 100 rather
    than 10^2).
    
    ``ax=None``
        The axis to set, or uses ``gca()`` if ``None``.
    ``freqs=None``
        Override the default frequency locations with your preferred tick
        locations.
        
    See also: :func:`log_frequency_yaxis_labels`.
    
    Note: with log scaled axes, it can be useful to call ``axis('tight')``
    before setting the ticks.
    '''
    if ax is None:
        ax = gca()
    return log_frequency_axis_labels(ax.yaxis, freqs=freqs)

def log_frequency_axis_labels(ax, freqs=None):
    if freqs is not None:
        ax.set_major_locator(FixedLocator(freqs))
        ax.set_major_formatter(ScalarFormatter())
        ax.set_minor_locator(NullLocator())
        return
    xmin, xmax = ax.get_view_interval()
    # we use the first of these ranges that the data fits within
    allowed_ranges = [[1, 2, 4, 8, 16, 32, 64],
                      [50, 75, 100, 150, 200, 300, 400],
                      [10, 20, 40, 80, 160, 320],
                      [1, 2, 4, 8, 16, 32, 64, 100, 200, 400, 800],
                      [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
                      [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000],
                      ]
    found = False
    for R in allowed_ranges:
        if xmin>=amin(R)*0.9999 and xmax<=amax(R)*1.0001:
            found = True
            break
    if not found:
        ax.set_major_locator(LogLocator(base=2))
    else:
        ax.set_major_locator(FixedLocator(R))
    ax.set_major_formatter(ScalarFormatter())
    ax.set_minor_locator(NullLocator())

if __name__=='__main__':
    for i, cfs in enumerate([erbspace(150*Hz, 5*kHz, 100),
                             erbspace(2*Hz, 64*Hz, 100),
                             erbspace(100*Hz, 10*kHz, 100),
                             erbspace(100*Hz, 400*Hz, 100),
                             ]):
        subplot(2, 2, i+1)
        #cfs = erbspace(150*Hz, 5*kHz, 100)
        semilogx(cfs, 1-((arange(len(cfs))-len(cfs)/2.0)/(len(cfs)/2.0))**2)
        axis('tight')
        log_frequency_xaxis_labels()
    show()
