#!/usr/bin/env python
'''
Interface example
Install cherrypy for this example
Then run the script and go to http://localhost:8080 on your web browser
You can use cherrypy to write html interfaces to your code.
'''

from brian import *
import cherrypy
import os.path

# The server is defined here
class MyInterface(object):
    @cherrypy.expose
    def index(self): # redirect to the html page we wrote
        return '<meta HTTP-EQUIV="Refresh" content="0;URL=index.html">'

    @cherrypy.expose
    def runscript(self, we="1.62", wi="-9", **kwd): # 'runscript' is the script name
        # we and wi are the names of form fields
        we = float(we)
        wi = float(wi)
        # From minimalexample
        reinit_default_clock()
        eqs = '''
        dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
        dge/dt = -ge/(5*ms) : volt
        dgi/dt = -gi/(10*ms) : volt
        '''
        P = NeuronGroup(4000, model=eqs, threshold= -50 * mV, reset= -60 * mV)
        P.v = -60 * mV + 10 * mV * rand(len(P))
        Pe = P.subgroup(3200)
        Pi = P.subgroup(800)
        Ce = Connection(Pe, P, 'ge')
        Ci = Connection(Pi, P, 'gi')
        Ce.connect_random(Pe, P, 0.02, weight=we * mV)
        Ci.connect_random(Pi, P, 0.02, weight=wi * mV)
        M = SpikeMonitor(P)
        run(.5 * second)
        clf()
        raster_plot(M)
        savefig('image.png')
        # Redirect to the html page we wrote
        return '<meta HTTP-EQUIV="Refresh" content="0;URL=results.html">'

# Set the directory for static files
current_dir = os.path.dirname(os.path.abspath(__file__))
conf = {'/': {'tools.staticdir.on':True,
              'tools.staticdir.dir':current_dir}}

# Start the server
cherrypy.quickstart(MyInterface(), config=conf)
