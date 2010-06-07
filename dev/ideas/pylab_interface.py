from brian import *
from matplotlib.widgets import Button

fig=figure()
plot([0, 0], [1, 1])

def onbutton(e):
    x, y=e.xdata, e.ydata
    print x, y

def nextbutton(e):
    print 'hello'

fig.canvas.mpl_connect('button_press_event', onbutton)

axnext=axes([0.81, 0.05, 0.1, 0.075])
bnext=Button(axnext, 'Next')
bnext.on_clicked(nextbutton)

show()
